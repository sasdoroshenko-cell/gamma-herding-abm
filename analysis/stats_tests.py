"""
Statistical tests of the project (task C-7; docs/prereg.md, section 6).

- Jonckheere-Terpstra test against an ordered alternative: own implementation, permutation p-value
  (all labellings when there are few of them, otherwise 10 000 random permutations) and the
  normal approximation with the tie correction for comparison;
- one-sided Mann-Whitney test (scipy) with P(X > Y) = U / (n1 n2);
- Holm's step-down correction;
- percentile bootstrap confidence intervals (each sample is resampled separately);
- Kendall's tau between the order of the configurations and the observed values.

The three hypotheses of the prereg are wrapped in main_hypothesis_test, volatility_test and hysteresis_test;
each returns a HypothesisReport with everything the report has to state about a test: H0 and H1, alpha,
statistic, p-value, Holm-adjusted p-value, effect sizes with CIs, sample sizes and a verbal conclusion.

All observations are one number per run (one J_c per replication, one sigma or one loop area per
run), never time steps (prereg, section 6).
"""
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial, isnan, nan, prod, sqrt

import numpy as np
import pandas as pd
from scipy import stats

from experiments import config as C

ORDER = ('short', 'none', 'long')


def _clean(groups) -> list:
    out = [np.asarray(g, dtype=float).ravel() for g in groups]
    if len(out) < 2:
        raise ValueError('at least two samples are needed')
    for g in out:
        if len(g) == 0:
            raise ValueError('empty sample')
        if np.isnan(g).any():
            raise ValueError('NaN in a sample: failed runs must be removed or replaced before testing')
    return out


def _rng(rng, stream: int):
    """Generator for one test: the given one, or a fixed stream of ANALYSIS_SEED, so results repeat."""
    return rng if rng is not None else np.random.default_rng([C.ANALYSIS_SEED, stream])


# ---------------------------------------------------------------- Jonckheere-Terpstra

def jt_statistic(groups) -> float:
    """J = sum over pairs of groups i < j of #(x_j > x_i) + 0.5 * #(x_j = x_i)."""
    groups = _clean(groups)
    total = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            d = groups[j][None, :] - groups[i][:, None]
            total += (d > 0).sum() + 0.5 * (d == 0).sum()
    return float(total)


def jt_null_moments(groups) -> tuple:
    """
    Mean and variance of J under H0, with the correction for ties (Hollander, Wolfe & Chicken,
    Nonparametric Statistical Methods, section 6.2):
        E J   = (N^2 - sum n_i^2) / 4
        Var J = [N(N-1)(2N+5) - sum n_i(n_i-1)(2n_i+5) - sum t_k(t_k-1)(2t_k+5)] / 72
              + [sum n_i(n_i-1)(n_i-2)] [sum t_k(t_k-1)(t_k-2)] / [36 N(N-1)(N-2)]
              + [sum n_i(n_i-1)] [sum t_k(t_k-1)] / [8 N(N-1)]
    where n_i are the sample sizes and t_k the sizes of groups of tied values.
    """
    groups = _clean(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    N = n.sum()
    _, t = np.unique(np.concatenate(groups), return_counts=True)
    t = t.astype(float)
    mean = (N ** 2 - (n ** 2).sum()) / 4
    var = (N * (N - 1) * (2 * N + 5) - (n * (n - 1) * (2 * n + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72
    if N > 2:
        var += (n * (n - 1) * (n - 2)).sum() * (t * (t - 1) * (t - 2)).sum() / (36 * N * (N - 1) * (N - 2))
    var += (n * (n - 1)).sum() * (t * (t - 1)).sum() / (8 * N * (N - 1))
    return float(mean), float(var)


def _labellings(sizes):
    """Every distinct assignment of group labels to positions 0..N-1 with the given group sizes."""
    def assign(free, k):
        if k == len(sizes) - 1:
            yield [(k, free)]
            return
        for chosen in combinations(free, sizes[k]):
            rest = tuple(p for p in free if p not in set(chosen))
            for tail in assign(rest, k + 1):
                yield [(k, chosen)] + tail

    N = sum(sizes)
    for parts in assign(tuple(range(N)), 0):
        labels = np.empty(N, dtype=int)
        for k, pos in parts:
            labels[list(pos)] = k
        yield labels


def _jt_of_labels(beats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """J for every row of `labels` (m x N); beats[a, b] = 1 if x_b > x_a, 0.5 if equal, 0 otherwise."""
    earlier = labels[:, :, None] < labels[:, None, :]
    return (earlier * beats).sum(axis=(1, 2))


@dataclass
class JTResult:
    statistic: float
    mean: float
    var: float
    z: float
    p_perm: float
    p_asymptotic: float
    n_perm: int
    exact: bool
    sizes: tuple


def jonckheere_terpstra(groups, n_perm: int = C.N_PERM, rng=None, max_exact: int = 20_000) -> JTResult:
    """
    One-sided test of H0 (all samples from one distribution) against the ordered alternative
    groups[0] <= groups[1] <= ... with at least one strict inequality. Large J speaks for H1.

    p_perm: if the number of distinct labellings N! / prod(n_i!) is at most max_exact, all of them are
    enumerated (exact p-value); otherwise (1 + #{J* >= J}) / (1 + n_perm) over random permutations.
    p_asymptotic: normal approximation with the tie-corrected variance, for comparison.
    """
    groups = _clean(groups)
    sizes = tuple(len(g) for g in groups)
    values = np.concatenate(groups)
    labels = np.repeat(np.arange(len(groups)), sizes)
    stat = jt_statistic(groups)
    mean, var = jt_null_moments(groups)
    z = (stat - mean) / sqrt(var) if var > 0 else nan
    p_asym = float(stats.norm.sf(z)) if not isnan(z) else nan

    beats = (values[None, :] > values[:, None]) + 0.5 * (values[None, :] == values[:, None])
    tol = 1e-9 * max(1.0, abs(stat))
    n_labellings = factorial(len(values)) // prod(factorial(s) for s in sizes)
    if n_labellings <= max_exact:
        dist = np.concatenate([_jt_of_labels(beats, np.array(chunk))
                               for chunk in _chunks(_labellings(list(sizes)), 2000)])
        return JTResult(stat, mean, var, z, float(np.mean(dist >= stat - tol)), p_asym, len(dist), True, sizes)

    rng = _rng(rng, 0)
    count, done = 0, 0
    while done < n_perm:
        m = min(500, n_perm - done)
        perm = rng.permuted(np.tile(labels, (m, 1)), axis=1)
        count += int((_jt_of_labels(beats, perm) >= stat - tol).sum())
        done += m
    return JTResult(stat, mean, var, z, (1 + count) / (1 + n_perm), p_asym, n_perm, False, sizes)


def _chunks(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ---------------------------------------------------------------- Mann-Whitney, Holm

@dataclass
class MWResult:
    u: float                 # #(x > y) + 0.5 #(x = y)
    p: float
    method: str
    prob_superiority: float  # U / (n1 n2) = P(X > Y) + 0.5 P(X = Y)
    n1: int
    n2: int


def mann_whitney_greater(x, y) -> MWResult:
    """One-sided Mann-Whitney test, H1: X is stochastically greater than Y. Exact p without ties."""
    x, y = _clean([x, y])
    d = x[:, None] - y[None, :]
    u = float((d > 0).sum() + 0.5 * (d == 0).sum())
    ties = len(np.unique(np.concatenate([x, y]))) < len(x) + len(y)
    method = 'asymptotic' if ties else 'exact'
    res = stats.mannwhitneyu(x, y, alternative='greater', method=method)
    return MWResult(u, float(res.pvalue), method, u / (len(x) * len(y)), len(x), len(y))


def holm(pvalues, alpha: float = C.ALPHA):
    """
    Holm's step-down procedure. Adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)) for the sorted
    p-values; a hypothesis is rejected when its adjusted p-value is <= alpha.
    Returns (adjusted p-values, rejected flags) in the original order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind='stable')):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted, adjusted <= alpha


# ---------------------------------------------------------------- effect sizes and bootstrap

def median_difference(x, y) -> float:
    return float(np.median(x) - np.median(y))


def median_ratio(x, y) -> float:
    return float(np.median(x) / np.median(y))


def prob_superiority(x, y) -> float:
    """P(X > Y) + 0.5 P(X = Y) over all pairs, i.e. U / (n1 n2)."""
    d = np.asarray(x, dtype=float)[:, None] - np.asarray(y, dtype=float)[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def kendall_tau_groups(*groups) -> float:
    """Kendall's tau-b between the group number (1, 2, 3, ...) and the observed values."""
    labels = np.concatenate([np.full(len(g), k + 1) for k, g in enumerate(groups)])
    return float(stats.kendalltau(labels, np.concatenate(groups)).statistic)


@dataclass
class Effect:
    name: str
    estimate: float
    low: float
    high: float


def bootstrap_ci(statistic, samples, n_boot: int = C.N_BOOT, level: float = C.CI_LEVEL, rng=None,
                 name: str = '') -> Effect:
    """Estimate and percentile bootstrap CI of statistic(*samples); every sample is resampled on its own."""
    samples = _clean(samples)
    rng = _rng(rng, 0)
    estimate = float(statistic(*samples))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = statistic(*(s[rng.integers(0, len(s), len(s))] for s in samples))
    low, high = np.nanpercentile(boot, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return Effect(name, estimate, float(low), float(high))


# ---------------------------------------------------------------- reports

@dataclass
class HypothesisReport:
    key: str
    quantity: str
    direction: str        # H1 in words, e.g. 'J_c(short) < J_c(none) < J_c(long)'
    h0: str
    h1: str
    test: str
    statistic_name: str
    statistic: float
    p_value: float
    p_note: str
    n: dict
    effects: list
    alpha: float = C.ALPHA
    p_holm: float = nan
    reject: bool = None
    extra: dict = field(default_factory=dict)
    ordered: bool = False   # Jonckheere-Terpstra: H1 is an ordering of three groups

    def conclusion(self) -> str:
        p = self.p_holm if not isnan(self.p_holm) else self.p_value
        which = 'с поправкой Холма' if not isnan(self.p_holm) else 'без поправки'
        if p <= self.alpha:
            text = (f"H₀ отвергается: p {which} = {p:.4g} ≤ α = {self.alpha}. "
                    f"Данные согласуются с H₁: {self.direction}.")
            if self.ordered:
                text += (" Тест подтверждает упорядоченный сдвиг в целом; какие из двух неравенств "
                         "строгие, показывают доверительные интервалы разностей (отношений) медиан.")
            return text
        return (f"H₀ не отвергается: p {which} = {p:.4g} > α = {self.alpha}. "
                f"По этим данным нельзя утверждать, что {self.direction}.")

    def to_markdown(self) -> str:
        n = ', '.join(f"{k}: {v}" for k, v in self.n.items())
        lines = [f"**{self.key}. {self.quantity}**", '',
                 f"- H₀: {self.h0}", f"- H₁: {self.h1}",
                 f"- Тест: {self.test}; α = {self.alpha}",
                 f"- Наблюдений (репликаций): {n}",
                 f"- Статистика: {self.statistic_name} = {self.statistic:.4g}",
                 f"- p-значение: {self.p_value:.4g} ({self.p_note})"]
        if not isnan(self.p_holm):
            lines.append(f"- p с поправкой Холма: {self.p_holm:.4g}")
        for k, v in self.extra.items():
            lines.append(f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}")
        for e in self.effects:
            lines.append(f"- {e.name}: {e.estimate:.4g}, ДИ 95% [{e.low:.4g}; {e.high:.4g}]")
        lines.append(f"- Вывод: {self.conclusion()}")
        return '\n'.join(lines)

    def to_row(self) -> dict:
        row = {'hypothesis': self.key, 'test': self.test, 'statistic': self.statistic, 'p': self.p_value,
               'p_holm': self.p_holm, 'reject': self.reject, **{f'n_{k}': v for k, v in self.n.items()}}
        for i, e in enumerate(self.effects, 1):
            row.update({f'effect{i}': e.name, f'effect{i}_est': e.estimate, f'effect{i}_low': e.low,
                        f'effect{i}_high': e.high})
        return row


def _samples(data: dict, keys) -> list:
    return [np.asarray(data[k], dtype=float) for k in keys]


def main_hypothesis_test(jc: dict, n_perm=C.N_PERM, n_boot=C.N_BOOT, rng=None) -> HypothesisReport:
    """Main hypothesis: J_c(short) <= J_c(none) <= J_c(long). jc maps configuration -> J_c of replications."""
    rng = _rng(rng, 1)
    s, n, l = _samples(jc, ORDER)
    jt = jonckheere_terpstra([s, n, l], n_perm, rng)
    effects = [
        bootstrap_ci(median_difference, [n, s], n_boot, rng=rng, name='med J_c(none) − med J_c(short)'),
        bootstrap_ci(median_difference, [l, n], n_boot, rng=rng, name='med J_c(long) − med J_c(none)'),
        bootstrap_ci(kendall_tau_groups, [s, n, l], n_boot, rng=rng,
                     name='τ Кендалла между порядком конфигурации (short = 1, none = 2, long = 3) и J_c'),
    ]
    return HypothesisReport(
        key='Г', quantity='критическая связь J_c = argmax_J χ(J)',
        direction='J_c(short) < J_c(none) < J_c(long)',
        h0='распределения J_c в трёх конфигурациях одинаковы',
        h1='J_c(short) ≤ J_c(none) ≤ J_c(long), хотя бы одно неравенство строгое',
        test='Джонкхира–Терпстры, односторонний', statistic_name='J', statistic=jt.statistic,
        p_value=jt.p_perm, p_note=_perm_note(jt), n={k: len(v) for k, v in zip(ORDER, (s, n, l))},
        effects=effects, extra={'z (нормальное приближение)': jt.z, 'p (нормальное приближение)': jt.p_asymptotic}, ordered=True)


def volatility_test(sigma: dict, n_perm=C.N_PERM, n_boot=C.N_BOOT, rng=None) -> HypothesisReport:
    """Hypothesis H1: sigma(long) <= sigma(none) <= sigma(short) at J = J_fix."""
    rng = _rng(rng, 2)
    s, n, l = _samples(sigma, ORDER)
    jt = jonckheere_terpstra([l, n, s], n_perm, rng)
    effects = [
        bootstrap_ci(median_ratio, [s, n], n_boot, rng=rng, name='med σ(short) / med σ(none)'),
        bootstrap_ci(median_ratio, [n, l], n_boot, rng=rng, name='med σ(none) / med σ(long)'),
    ]
    return HypothesisReport(
        key='В1', quantity='σ лог-доходностей при J = J_fix',
        direction='σ(long) < σ(none) < σ(short)',
        h0='распределения σ в трёх конфигурациях одинаковы',
        h1='σ(long) ≤ σ(none) ≤ σ(short), хотя бы одно неравенство строгое',
        test='Джонкхира–Терпстры, односторонний', statistic_name='J', statistic=jt.statistic,
        p_value=jt.p_perm, p_note=_perm_note(jt), n={k: len(v) for k, v in zip(ORDER, (s, n, l))},
        effects=effects, extra={'z (нормальное приближение)': jt.z, 'p (нормальное приближение)': jt.p_asymptotic}, ordered=True)


def hysteresis_test(area: dict, n_boot=C.N_BOOT, rng=None) -> HypothesisReport:
    """Hypothesis H2: the loop area A(short) is stochastically greater than A(none)."""
    rng = _rng(rng, 3)
    s, n = _samples(area, ('short', 'none'))
    mw = mann_whitney_greater(s, n)
    effects = [
        bootstrap_ci(prob_superiority, [s, n], n_boot, rng=rng, name='P(A_short > A_none) = U/(n₁n₂)'),
        bootstrap_ci(median_difference, [s, n], n_boot, rng=rng, name='med A(short) − med A(none)'),
    ]
    return HypothesisReport(
        key='В2', quantity='площадь петли гистерезиса A',
        direction='A(short) стохастически больше A(none)',
        h0='распределения A в конфигурациях short и none одинаковы',
        h1='A(short) стохастически больше A(none)',
        test='Манна–Уитни, односторонний', statistic_name='U', statistic=mw.u, p_value=mw.p,
        p_note='точное' if mw.method == 'exact' else 'нормальное приближение с поправкой на связки',
        n={'short': mw.n1, 'none': mw.n2}, effects=effects)


def _perm_note(jt: JTResult) -> str:
    if jt.exact:
        return f'точное, по всем {jt.n_perm} перестановкам меток'
    return f'перестановочное, {jt.n_perm} перестановок'


def apply_holm(reports, alpha: float = C.ALPHA):
    """Holm correction over the family of reports (the three main tests); fills p_holm and reject."""
    adjusted, reject = holm([r.p_value for r in reports], alpha)
    for r, p, rej in zip(reports, adjusted, reject):
        r.p_holm, r.reject = float(p), bool(rej)
    return reports


def results_table(reports) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in reports])


def describe_samples(samples: dict) -> pd.DataFrame:
    """
    Descriptive statistics for checking the assumptions (prereg, section 6): the rank tests compare
    locations cleanly only if the samples have a similar shape, so their spread and skewness are
    shown next to each other; the Fligner-Killeen test of equal spread is a numeric complement to the
    box plots, not a gate for the main tests.
    """
    rows = []
    for k, v in samples.items():
        v = np.asarray(v, dtype=float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append({'sample': k, 'n': len(v), 'median': med, 'iqr': q3 - q1, 'mean': v.mean(),
                     'sd': v.std(ddof=1) if len(v) > 1 else nan,
                     'skewness': float(stats.skew(v)) if len(v) > 2 else nan})
    out = pd.DataFrame(rows)
    values = [np.asarray(v, dtype=float) for v in samples.values()]
    if len(values) > 1 and all(len(v) > 1 for v in values):
        out.attrs['fligner_p'] = float(stats.fligner(*values).pvalue)
    return out
