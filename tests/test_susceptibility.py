"""
Checks for analysis/susceptibility.py: J_c on synthetic chi(J) curves with a known maximum and the
replacement rules of docs/prereg.md, section 7.
"""
import numpy as np
import pandas as pd
import pytest

from analysis import susceptibility as sus
from experiments import config as C

GRID = np.array(C.J_GRID)


def chi_curve(jc, width=0.4, height=5.0):
    return height - (GRID - jc) ** 2 / width   # a parabola: its vertex is the true J_c


@pytest.mark.parametrize("jc", [0.73, 1.05, 0.31, 1.88])
def test_parabola_recovers_a_known_maximum(jc):
    p = sus.critical_coupling(GRID, chi_curve(jc))
    assert p.refined and not p.edge
    assert p.jc == pytest.approx(jc)


def test_gaussian_peak_is_recovered_within_a_fraction_of_the_grid_step():
    true = 0.86
    chi = np.exp(-(GRID - true) ** 2 / (2 * 0.3 ** 2))
    p = sus.critical_coupling(GRID, chi)
    assert abs(p.jc - true) < 0.02          # grid step 0.1, the argmax alone would be off by 0.04
    assert p.j_argmax == pytest.approx(0.9)


def test_vertex_with_uneven_spacing():
    x = [0.5, 0.7, 0.8]
    y = [-(v - 0.66) ** 2 for v in x]
    assert sus.parabolic_vertex(x, y) == pytest.approx(0.66)


def test_flat_triple_returns_the_middle_point():
    assert sus.parabolic_vertex([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) == pytest.approx(0.2)


@pytest.mark.parametrize("peak,edge_J", [(-0.5, 0.0), (2.6, 2.0)])
def test_peak_at_the_edge_of_the_grid_is_flagged(peak, edge_J):
    p = sus.critical_coupling(GRID, chi_curve(peak))
    assert p.edge and not p.refined
    assert p.jc == edge_J


def test_failed_points_are_ignored():
    chi = chi_curve(1.0)
    chi[10] = np.nan          # the run at the true maximum failed
    p = sus.critical_coupling(GRID, chi)
    assert p.refined and p.j_argmax in (0.9, 1.1)
    assert p.jc == pytest.approx(1.0)  # neighbours 0.9 / 1.1 and their other neighbour still fit the parabola


def sweep_frame(reps_failed: dict, configs=('short', 'none', 'long'), jc=1.0):
    """Fake sweep table: every replication has chi = parabola with vertex jc; listed points fail."""
    rows = []
    for config in configs:
        for rep, failed in reps_failed.items():
            for j, J in enumerate(C.J_GRID):
                bad = j in failed
                rows.append({'config': config, 'size_index': C.BASE_SIZE, 'j_index': j, 'rep': rep, 'J': J,
                             'failed_at': 100.0 if bad else np.nan, 'chi': np.nan if bad else chi_curve(jc)[j],
                             'mean_abs_m': min(1.0, J / 1.25), 'sigma': 0.01 + 0.001 * rep, 'seed': rep})
    df = pd.DataFrame(rows)
    df['failed'] = df['failed_at'].notna()
    return df


def test_replication_with_more_than_three_failed_points_is_replaced_by_the_next():
    df = sweep_frame({0: set(), 1: {1, 2, 3, 4}, 2: {5, 6, 7}, 3: set()})
    jc = sus.jc_table(df, n_reps=3)
    for _, g in jc.groupby('config'):
        assert list(g['rep']) == [0, 2, 3]          # rep 1 (4 failed points) is replaced by rep 3
    report = sus.failure_report(df, jc)
    assert set(report['replaced_reps']) == {1}
    assert set(report['failed_runs']) == {7}
    assert np.allclose(jc['jc'], 1.0)


def test_incomplete_replication_is_not_used():
    df = sweep_frame({0: set(), 1: set()})
    df = df[~((df['rep'] == 1) & (df['j_index'] == 20))]    # rep 1 has not finished its last point
    with pytest.warns(UserWarning, match="only 1 valid replications"):
        jc = sus.jc_table(df, n_reps=30)
    assert set(jc["rep"]) == {0}


def test_protocol_points_follow_the_prereg_rules():
    df = sweep_frame({0: set()}, configs=('none',))      # mean |m| = J / 1.25, capped at 1
    assert sus.protocol_point(df, 'fix') == pytest.approx(1.0)   # first J with J / 1.25 >= 0.8 (exactly 0.8 counts)
    # chi = 5 - (J - 1)^2 / 0.4: half of the maximum is reached only at J = 0
    assert sus.protocol_point(df, 'ref') == pytest.approx(0.0)


def none_frame(chi_of_J):
    rows = [{'config': 'none', 'J': J, 'chi': chi_of_J(J), 'mean_abs_m': 0.5, 'failed_at': np.nan, 'seed': s}
            for J in C.J_GRID for s in range(3)]
    df = pd.DataFrame(rows)
    df['failed'] = False
    return df


def test_j_ref_is_the_last_point_below_half_of_the_chi_peak():
    df = none_frame(lambda J: np.exp(-(J - 1.2) ** 2 / (2 * 0.3 ** 2)))   # half maximum at J = 0.85
    assert sus.protocol_point(df, 'ref') == pytest.approx(0.8)


def test_j_ref_falls_back_to_the_start_of_the_grid():
    df = none_frame(lambda J: np.exp(-(J - 0.2) ** 2 / (2 * 0.5 ** 2)))   # chi above half everywhere left of the peak
    assert sus.protocol_point(df, 'ref') == pytest.approx(0.0)


def test_volatility_sample_takes_the_first_finished_runs_at_j_fix():
    df = sweep_frame({0: set(), 1: {11}, 2: set(), 3: set()})
    s = sus.volatility_sample(df, j_fix=1.1, n_reps=3)
    for _, g in s.groupby('config'):
        assert list(g['rep']) == [0, 2, 3]       # rep 1 failed at J = 1.1


def test_peak_next_to_a_failed_run_is_flagged():
    chi = chi_curve(0.02)                 # true peak just above J = 0
    chi[0] = np.nan                       # the run at J = 0 failed: the maximum is the first finished point
    p = sus.critical_coupling(GRID, chi)
    assert p.edge and p.neighbour_failed and not p.refined
    assert p.jc == pytest.approx(0.1)


def test_jc_table_takes_one_book_size():
    main = sweep_frame({0: set(), 1: set()})
    other = main.assign(size_index=C.BASE_SIZE + 1)
    both = pd.concat([main, other], ignore_index=True)
    jc = sus.jc_table(both, n_reps=2)
    assert set(jc['size_index']) == {C.BASE_SIZE} and len(jc) == 3 * 2
    assert len(sus.jc_table(both, n_reps=2, size_index=None)) == 3 * 2 * 2


def test_mixed_models_in_one_file_are_refused(tmp_path):
    df = sweep_frame({0: set()}).assign(model=['a'] * 63)
    df.loc[0, 'model'] = 'b'
    path = tmp_path / 'sweep.csv'
    df.drop(columns='failed').to_csv(path, index=False)
    with pytest.raises(ValueError):
        sus.load_sweep(path)
