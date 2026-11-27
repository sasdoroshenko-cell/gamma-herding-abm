"""
Susceptibility chi(J) and the critical coupling J_c of every replication (task C-5; docs/prereg.md,
sections 5 and 7).

    chi(J) = N_ch * Var_t(|m_t|)   over steps 501-1500 of the run at coupling J
    J_c    = argmax_J chi(J), refined by the vertex of the parabola through the maximum and its two
             neighbours when the maximum is inside the grid

Rules of the prereg applied here:
- a failed run leaves its point out of chi(J) of that replication;
- a replication with more than 3 failed points is replaced by the next one in the seed stream,
  so the sample is the first 30 valid replications in the order of their numbers;
- a maximum at the edge of the grid (J = 0 or J = 2) gives J_c equal to the edge and is flagged.

Input is the table written by experiments/run_sweep.py (one row per run).
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments import config as C


def load_sweep(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'config': str, 'model': str, 'error': str})
    df['failed'] = df['failed_at'].notna()
    if 'model' in df and df['model'].nunique() > 1:
        raise ValueError(f"{path} mixes runs of different models: {sorted(df['model'].unique())}")
    return df


def parabolic_vertex(x, y) -> float:
    """
    Abscissa of the vertex of the parabola through three points (x0, y0), (x1, y1), (x2, y2) with
    x0 < x1 < x2 (the spacing may be uneven when a neighbouring run failed). If y1 is the largest of
    the three, the vertex lies in [x0, x2]; for a degenerate (flat) triple x1 is returned.
    """
    x0, x1, x2 = map(float, x)
    y0, y1, y2 = map(float, y)
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if a >= 0:   # not a maximum (flat or convex)
        return x1
    return float(min(max(-b / (2 * a), x0), x2))


@dataclass
class Peak:
    jc: float          # estimate of J_c
    j_argmax: float    # grid point with the largest chi
    chi_max: float
    edge: bool         # maximum at the first or last finished point (grid edge, or its neighbour failed)
    refined: bool      # J_c is the vertex of a parabola, not a grid point
    neighbour_failed: bool = False   # flagged because a neighbouring run failed, not at the grid edge


def critical_coupling(J, chi, grid_min=C.J_GRID[0], grid_max=C.J_GRID[-1]) -> Peak:
    """J_c of one replication from its chi(J); NaN points (failed runs) are ignored."""
    J, chi = np.asarray(J, dtype=float), np.asarray(chi, dtype=float)
    keep = ~np.isnan(chi)
    J, chi = J[keep], chi[keep]
    if len(J) == 0:
        raise ValueError('no finished runs in the replication')
    order = np.argsort(J)
    J, chi = J[order], chi[order]
    i = int(np.argmax(chi))    # the first maximum if several are equal
    j_max = float(J[i])
    grid_edge = abs(j_max - grid_min) < 1e-9 or abs(j_max - grid_max) < 1e-9
    if grid_edge or i == 0 or i == len(J) - 1:
        # J_c cannot be refined: at the edge of the grid (prereg, section 7), or a neighbouring run failed;
        # both are flagged as edge peaks, so they are counted and left out in the exploratory recheck
        return Peak(j_max, j_max, float(chi[i]), True, False, neighbour_failed=not grid_edge)
    jc = parabolic_vertex(J[i - 1:i + 2], chi[i - 1:i + 2])
    return Peak(jc, j_max, float(chi[i]), False, True)


def replication_table(df: pd.DataFrame, n_grid: int = None, max_failed: int = C.MAX_FAILED_POINTS) -> pd.DataFrame:
    """
    One row per (config, size_index, rep): number of points run and failed, and the status
    'valid', 'replaced' (> max_failed failed points) or 'incomplete' (not every point of the grid run).
    """
    n_grid = n_grid or df['j_index'].nunique()
    g = df.groupby(['config', 'size_index', 'rep'])
    t = pd.DataFrame({'n_points': g['j_index'].nunique(), 'n_failed': g['failed'].sum()}).reset_index()
    t['status'] = np.where(t['n_points'] < n_grid, 'incomplete',
                           np.where(t['n_failed'] > max_failed, 'replaced', 'valid'))
    return t


def jc_table(df: pd.DataFrame, n_reps: int = C.N_REPS, max_failed: int = C.MAX_FAILED_POINTS,
             grid=None, size_index=C.BASE_SIZE) -> pd.DataFrame:
    """
    J_c of the replications that enter the test: for every configuration the first n_reps valid
    replications by number, at book size `size_index` (the main experiment by default; None for
    every size, e.g. the sensitivity analysis, which then has to be split by size_index).
    Columns: config, size_index, rep, jc, j_argmax, chi_max, edge, refined, neighbour_failed, n_failed.
    Warns if fewer than n_reps valid replications are available.
    """
    grid = list(C.J_GRID) if grid is None else list(grid)
    if size_index is not None:
        df = df[df['size_index'] == size_index]
    reps = replication_table(df, len(grid), max_failed)
    rows = []
    for (config, size), r in reps.groupby(['config', 'size_index']):
        used = r[r['status'] == 'valid'].sort_values('rep').head(n_reps)
        if len(used) < n_reps:
            warnings.warn(f"{config}, size {size}: only {len(used)} valid replications of {n_reps}")
        for _, rep in used.iterrows():
            runs = df[(df['config'] == config) & (df['size_index'] == size) & (df['rep'] == rep['rep'])]
            chi = runs['chi'].where(~runs['failed'])
            p = critical_coupling(runs['J'], chi, grid[0], grid[-1])
            rows.append({'config': config, 'size_index': size, 'rep': rep['rep'], 'jc': p.jc,
                         'j_argmax': p.j_argmax, 'chi_max': p.chi_max, 'edge': p.edge, 'refined': p.refined,
                         'neighbour_failed': p.neighbour_failed, 'n_failed': rep['n_failed']})
    return pd.DataFrame(rows)


def failure_report(df: pd.DataFrame, jc: pd.DataFrame = None, n_grid: int = None,
                   max_failed: int = C.MAX_FAILED_POINTS) -> pd.DataFrame:
    """Per (config, size_index): runs, failed runs, replaced replications, edge peaks (prereg, section 7)."""
    reps = replication_table(df, n_grid, max_failed)
    out = df.groupby(['config', 'size_index']).agg(runs=('failed', 'size'), failed_runs=('failed', 'sum'))
    out['replaced_reps'] = reps[reps['status'] == 'replaced'].groupby(['config', 'size_index']).size()
    if jc is not None and len(jc):
        out['used_reps'] = jc.groupby(['config', 'size_index']).size()
        out['edge_peaks'] = jc.groupby(['config', 'size_index'])['edge'].sum()
    return out.fillna(0).astype(int).reset_index()


def mean_curves(df: pd.DataFrame, value: str, jc: pd.DataFrame, size_index: int) -> pd.DataFrame:
    """Mean and 10-90% band of `value` (e.g. 'chi' or 'mean_abs_m') over the replications used for J_c."""
    used = jc[jc['size_index'] == size_index][['config', 'rep']]
    d = df[(df['size_index'] == size_index) & ~df['failed']].merge(used, on=['config', 'rep'])
    g = d.groupby(['config', 'J'])[value]
    return pd.DataFrame({'mean': g.mean(), 'q10': g.quantile(0.1), 'q90': g.quantile(0.9),
                         'n': g.size()}).reset_index()


def protocol_point(df_none: pd.DataFrame, rule: str) -> float:
    """
    J_fix or J_ref from runs without dealers, using the mean over seeds at every grid point:
      'fix' (prereg, section 4): the smallest grid point with mean |m| >= 0.8, NaN if there is none;
      'ref' (amendment 1): the largest grid point to the left of the peak of the mean chi(J) where chi
            is at most half of its maximum, i.e. a point clearly below the transition; the smallest
            grid point if there is none. The old rule (mean |m| <= 0.3) selects no point when the
            price channel is strong, because |m| is above 0.3 already at J = 0.
    """
    d = df_none[~df_none['failed']]
    if rule == 'fix':
        m = d.groupby('J')['mean_abs_m'].mean().sort_index()
        hits = m[m >= C.M_FIX_LEVEL]
        return float(hits.index.min()) if len(hits) else float('nan')
    if rule == 'ref':
        chi = d.groupby('J')['chi'].mean().sort_index()
        left = chi[chi.index < chi.idxmax()]
        hits = left[left <= C.REF_CHI_FRACTION * chi.max()]
        return float(hits.index.max()) if len(hits) else float(chi.index.min())
    raise ValueError("rule must be 'fix' or 'ref'")


def volatility_sample(df: pd.DataFrame, j_fix: float, size_index: int = C.BASE_SIZE,
                      n_reps: int = C.N_REPS, column: str = 'sigma') -> pd.DataFrame:
    """
    Sample for hypothesis H1: sigma of the first n_reps finished runs at J = J_fix for every
    configuration, in the order of replication numbers (a failed run is replaced by the next seed).
    column='sigma' is the one-step sigma of the prereg; f'sigma_{C.SIGMA_HORIZON}' the secondary analysis.
    """
    d = df[(df['size_index'] == size_index) & (np.abs(df['J'] - j_fix) < 1e-9) & ~df['failed']]
    out = (d.sort_values('rep').groupby('config', group_keys=False).head(n_reps)
           [['config', 'rep', 'seed', column]].reset_index(drop=True))
    _warn_short(out, n_reps, 'J_fix')
    return out


def _warn_short(sample: pd.DataFrame, n_reps: int, what: str):
    for config, n in sample.groupby('config').size().items():
        if n < n_reps:
            warnings.warn(f"{config}: only {n} finished runs of {n_reps} ({what})")


def load_hysteresis(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'config': str, 'grid': str, 'model': str, 'error': str})
    df['failed'] = df['failed_at'].notna()
    if 'model' in df and df['model'].nunique() > 1:
        raise ValueError(f"{path} mixes loops of different models: {sorted(df['model'].unique())}")
    return df


def hysteresis_sample(df_h: pd.DataFrame, size_index: int = C.BASE_SIZE, n_reps: int = C.N_REPS) -> pd.DataFrame:
    """
    Sample for hypothesis H2: loop area A of the first n_reps finished hysteresis runs of every
    configuration, in the order of replication numbers (a failed run is replaced by the next seed).
    """
    d = df_h[(df_h['size_index'] == size_index) & ~df_h['failed']]
    out = (d.sort_values('rep').groupby('config', group_keys=False).head(n_reps)
           [['config', 'rep', 'seed', 'area']].reset_index(drop=True))
    _warn_short(out, n_reps, 'hysteresis')
    return out
