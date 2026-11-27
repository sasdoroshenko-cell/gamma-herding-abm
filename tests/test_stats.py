"""
Checks for analysis/stats_tests.py on data with a known answer.
"""
import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analysis import stats_tests as st


def brute_force_jt(groups):
    total = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for a in groups[i]:
                for b in groups[j]:
                    total += 1.0 if b > a else 0.5 if b == a else 0.0
    return total


def exact_null_distribution(groups):
    values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    beats = (values[None, :] > values[:, None]) + 0.5 * (values[None, :] == values[:, None])
    return np.concatenate([st._jt_of_labels(beats, np.array(c))
                           for c in st._chunks(st._labellings([len(g) for g in groups]), 2000)])


# ---------------------------------------------------------------- Jonckheere-Terpstra

def test_jt_statistic_matches_brute_force_with_ties():
    groups = [[1, 2, 2, 5], [2, 3, 3], [1, 4, 6, 6, 2]]
    assert st.jt_statistic(groups) == brute_force_jt(groups)


def test_jt_perfect_order_has_exact_p_one_over_number_of_labellings():
    r = st.jonckheere_terpstra([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert r.statistic == 27          # every one of the 3 * 9 cross pairs is in order
    assert r.exact and r.n_perm == 1680   # 9! / (3! 3! 3!)
    assert r.p_perm == pytest.approx(1 / 1680)


def test_jt_reversed_order_gives_p_one():
    r = st.jonckheere_terpstra([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    assert r.statistic == 0 and r.p_perm == pytest.approx(1.0)


@pytest.mark.parametrize("groups", [
    [[1.0, 2.0, 3.0, 4.0], [2.5, 5.0, 6.0], [0.5, 7.0, 8.0, 9.0]],    # no ties
    [[1, 1, 2, 3], [2, 2, 3], [3, 4, 4, 1]],                            # ties within and across groups
])
def test_jt_null_mean_and_variance_equal_the_exact_permutation_moments(groups):
    dist = exact_null_distribution(groups)
    mean, var = st.jt_null_moments(groups)
    assert mean == pytest.approx(dist.mean())
    assert var == pytest.approx(dist.var())


def test_jt_variance_without_ties_reduces_to_the_textbook_formula():
    rng = np.random.default_rng(0)
    groups = [rng.normal(size=n) for n in (5, 7, 9)]
    n = np.array([5, 7, 9])
    N = n.sum()
    textbook = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    assert st.jt_null_moments(groups)[1] == pytest.approx(textbook)


def test_jt_permutation_and_asymptotic_p_agree_for_30_per_group():
    rng = np.random.default_rng(1)
    groups = [rng.normal(0.0, 1, 30), rng.normal(0.25, 1, 30), rng.normal(0.5, 1, 30)]
    r = st.jonckheere_terpstra(groups, n_perm=10_000, rng=np.random.default_rng(2))
    assert not r.exact
    assert abs(r.p_perm - r.p_asymptotic) < 0.01


def test_jt_detects_an_ordered_shift_and_not_a_reversed_one():
    rng = np.random.default_rng(3)
    groups = [rng.normal(0, 1, 30), rng.normal(1, 1, 30), rng.normal(2, 1, 30)]
    assert st.jonckheere_terpstra(groups, rng=np.random.default_rng(4)).p_perm < 0.001
    assert st.jonckheere_terpstra(groups[::-1], rng=np.random.default_rng(4)).p_perm > 0.99


def test_jt_false_positive_rate_is_about_alpha_under_h0():
    rng = np.random.default_rng(5)
    p = [st.jonckheere_terpstra([rng.normal(size=30) for _ in range(3)], n_perm=1).p_asymptotic
         for _ in range(1000)]
    assert 0.03 < np.mean(np.array(p) <= 0.05) < 0.07


def test_jt_rejects_nan():
    with pytest.raises(ValueError):
        st.jonckheere_terpstra([[1, 2], [3, np.nan], [4, 5]])


# ---------------------------------------------------------------- Mann-Whitney, Holm

def test_mann_whitney_u_and_probability_of_superiority():
    x, y = [5, 6, 7, 8], [1, 2, 3]
    r = st.mann_whitney_greater(x, y)
    assert r.u == 12 and r.prob_superiority == 1.0
    assert r.method == 'exact'
    assert r.p == pytest.approx(1 / 35)     # one of C(7, 3) = 35 equally likely rankings


def test_mann_whitney_counts_ties_as_one_half_like_scipy():
    x, y = [1, 2, 2, 3, 5], [2, 2, 4, 1]
    r = st.mann_whitney_greater(x, y)
    assert r.u == pytest.approx(stats.mannwhitneyu(x, y, alternative='greater').statistic)
    assert r.prob_superiority == pytest.approx(st.prob_superiority(x, y))
    assert r.method == 'asymptotic'


@pytest.mark.parametrize("p", [[0.01, 0.04, 0.03, 0.2], [0.001, 0.02, 0.04], [0.5, 0.01, 0.03]])
def test_holm_matches_statsmodels(p):
    adjusted, reject = st.holm(p, 0.05)
    expected_reject, expected_adjusted, _, _ = multipletests(p, alpha=0.05, method='holm')
    assert np.allclose(adjusted, expected_adjusted)
    assert list(reject) == list(expected_reject)


def test_holm_three_tests_by_hand():
    adjusted, reject = st.holm([0.02, 0.01, 0.04])
    assert np.allclose(adjusted, [0.04, 0.03, 0.04])   # 3 * 0.01, 2 * 0.02, max(0.04, 1 * 0.04)
    assert list(reject) == [True, True, True]


# ---------------------------------------------------------------- effects and bootstrap

def test_kendall_tau_is_one_for_separated_ordered_groups():
    tau = st.kendall_tau_groups([1, 2], [3, 4], [5, 6])
    assert tau == pytest.approx(stats.kendalltau([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6]).statistic)
    assert tau > 0.8


def test_bootstrap_ci_covers_a_known_shift():
    rng = np.random.default_rng(6)
    x, y = rng.normal(1.0, 1, 200), rng.normal(0.0, 1, 200)
    e = st.bootstrap_ci(st.median_difference, [x, y], n_boot=2000, rng=np.random.default_rng(7))
    assert e.low < 1.0 < e.high
    assert e.low < e.estimate < e.high


def test_bootstrap_ci_coverage_is_close_to_95_percent():
    rng = np.random.default_rng(8)
    hits = 0
    for _ in range(200):
        x, y = rng.exponential(1, 40), rng.exponential(1, 40) + 0.5   # true median difference -0.5
        e = st.bootstrap_ci(st.median_difference, [x, y], n_boot=500, rng=rng)
        hits += e.low <= -0.5 <= e.high
    assert 0.88 <= hits / 200 <= 0.99


def test_bootstrap_is_reproducible_with_the_default_seed():
    x, y = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0]
    a = st.bootstrap_ci(st.median_difference, [x, y], n_boot=300)
    b = st.bootstrap_ci(st.median_difference, [x, y], n_boot=300)
    assert (a.low, a.high) == (b.low, b.high)


# ---------------------------------------------------------------- hypothesis reports

def test_reports_of_the_three_hypotheses_with_holm():
    rng = np.random.default_rng(9)
    jc = {'short': rng.normal(0.6, 0.1, 30), 'none': rng.normal(0.8, 0.1, 30), 'long': rng.normal(1.0, 0.1, 30)}
    sigma = {'short': rng.normal(1.2, 0.1, 30), 'none': rng.normal(1.0, 0.1, 30), 'long': rng.normal(1.0, 0.1, 30)}
    area = {'short': rng.normal(0.3, 0.1, 30), 'none': rng.normal(0.3, 0.1, 30)}
    reports = st.apply_holm([st.main_hypothesis_test(jc, n_boot=500), st.volatility_test(sigma, n_boot=500),
                             st.hysteresis_test(area, n_boot=500)])
    main, vol, hyst = reports
    assert main.reject and main.p_value < 0.001
    assert main.effects[0].low > 0 and main.effects[1].low > 0      # both medians in the expected order
    assert vol.reject and vol.effects[0].estimate > 1
    assert hyst.p_holm >= hyst.p_value
    for r in reports:
        text = r.to_markdown()
        assert 'H₀' in text and 'ДИ 95%' in text and 'Холма' in text
    table = st.results_table(reports)
    assert list(table['hypothesis']) == ['Г', 'В1', 'В2']


def test_describe_samples_reports_spread_and_fligner():
    d = st.describe_samples({'a': [1, 2, 3, 4], 'b': [2, 4, 6, 8]})
    assert list(d['n']) == [4, 4] and d['iqr'][1] == pytest.approx(2 * d['iqr'][0])
    assert 0 <= d.attrs['fligner_p'] <= 1
