"""
Checks for experiments/config.py, measures.py and run_sweep.py: seeds, reproducibility, resume after
a crash, the replacement rules of docs/prereg.md (section 7) and the measurements written for every run.
Runs are short (a few hundred steps) so the whole file takes seconds.
"""
import random

import numpy as np
import pytest

from AgentBasedModel.agents import ExchangeAgent
from AgentBasedModel.builder import build_market
from AgentBasedModel.simulator import Simulator
from AgentBasedModel.utils import Order, OrderList
from experiments import config as C
from experiments import run_sweep as rs
from experiments.measures import fixed_volume_spread


# ---------------------------------------------------------------- config and seeds

def test_seeds_are_unique_across_configs_sizes_points_and_replications():
    seeds = [C.seed(c, s, j, r) for c in C.CONFIGS for s in range(len(C.SIZE_FACTORS))
             for j in list(range(len(C.J_GRID))) + [C.HYST_J_INDEX] for r in range(C.MAX_REP + 1)]
    assert len(seeds) == len(set(seeds))
    debug = {C.seed(c, 0, j, r, debug=True) for c in C.CONFIGS for j in range(len(C.J_GRID)) for r in range(10)}
    assert not debug & set(seeds)


def test_seed_formula_of_the_prereg():
    assert C.seed('long', 2, 11, 7) == 3 * 1_000_000 + 2 * 10_000 + 11 * 100 + 7
    with pytest.raises(ValueError):
        C.seed('long', 0, 0, C.MAX_REP + 1)


def test_protocol_values_of_the_prereg():
    assert C.J_GRID[0] == 0.0 and C.J_GRID[-1] == 2.0 and len(C.J_GRID) == 21
    assert (C.STEPS, C.BURN_IN, C.N_REPS) == (1500, 500, 30)
    assert C.STEPS - C.BURN_IN >= 1000
    assert C.SIZE_FACTORS[C.BASE_SIZE] == 1.0
    assert C.BASE.a1 == 0.6
    assert C.HYST_BURN_IN + 2 * len(C.J_GRID) * C.HYST_STEP == 8900


def test_market_config_and_book_size():
    assert C.book_size('none', 4, 288) == 0.0
    assert C.book_size('short', 0, 288) == 72.0
    cfg = C.market_config('long', 0.7, 144.0)
    assert (cfg.J, cfg.dealer_side, cfg.dealer_book, cfg.a1) == (0.7, 'long', 144.0, 0.6)


def test_full_runs_need_frozen_parameters(monkeypatch):
    monkeypatch.setattr(C, 'N0', None)
    with pytest.raises(RuntimeError):
        C.require_frozen()
    monkeypatch.setattr(C, 'N0', 288)
    monkeypatch.setattr(C, 'J_FIX', 1.1)
    C.require_frozen()


# ---------------------------------------------------------------- one run

def small_task(config='short', j=8, rep=0, steps=300, burn_in=100):
    return rs.make_task(config, C.BASE_SIZE, j, rep, n0=288, steps=steps, burn_in=burn_in, debug=True)


def test_same_seed_gives_the_same_run():
    a, b = rs.simulate(small_task()), rs.simulate(small_task())
    for k in rs.FIELDS:
        if k != 'runtime_s':
            assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k   # equal or both NaN


def test_measuring_between_chunks_does_not_change_the_simulation():
    task = small_task(steps=250, burn_in=50)
    whole = build_market(C.market_config(task.config, task.J, task.book), seed=task.seed)
    whole.simulator.simulate(250, silent=True)
    chunked = build_market(C.market_config(task.config, task.J, task.book), seed=task.seed)
    chunked.simulator.simulate(50, silent=True)
    for _ in range(20):
        fixed_volume_spread(chunked.simulator.exchange, 20)
        chunked.simulator.simulate(10, silent=True)
    assert whole.simulator.info.prices == chunked.simulator.info.prices
    # agent ids grow from run to run, so compare the sentiments in agent order, not the dict keys
    assert ([list(s.values()) for s in whole.simulator.info.sentiments]
            == [list(s.values()) for s in chunked.simulator.info.sentiments])


def test_run_row_has_the_journal_of_the_plan():
    row = rs.simulate(small_task())
    assert row['failed_at'] == ''
    assert row['chi'] == pytest.approx(35 * row['var_abs_m'])
    assert 0 <= row['mean_abs_m'] <= 1 and row['sigma'] > 0
    assert row['fvs_50'] >= row['fvs_10'] >= row['spread'] - 1e-9 > 0
    assert 0 < row['dealer_share'] < 1 and row['dealer_gamma'] < 0      # short book: negative gamma


def test_empty_book_dealers_do_not_trade():
    row = rs.simulate(small_task(config='none'))
    assert row['book'] == 0 and row['dealer_share'] == 0 and row['dealer_gamma'] == 0


def test_failed_run_is_written_not_dropped(monkeypatch):
    def broken(self, n_iter, silent=False):
        raise Exception('Price cannot be determined')
    monkeypatch.setattr(Simulator, 'simulate', broken)
    row = rs.simulate(small_task())
    assert row['failed_at'] == 0 and row['chi'] != row['chi']    # step of failure, statistics NaN


# ---------------------------------------------------------------- CSV, resume, replacements

def test_resume_skips_finished_runs(tmp_path):
    path = tmp_path / 'sweep.csv'
    tasks = [small_task(j=j, rep=r, steps=120, burn_in=40) for j in (0, 10) for r in (0, 1)]
    assert len(rs.run_tasks(tasks[:2], str(path), workers=1, progress=False)) == 2
    assert len(rs.run_tasks(tasks, str(path), workers=1, progress=False)) == 2   # only the two new ones
    assert len(rs.run_tasks(tasks, str(path), workers=1, progress=False)) == 0
    rows = rs.read_rows(str(path))
    assert sorted((r['j_index'], r['rep']) for r in rows) == [(0, 0), (0, 1), (10, 0), (10, 1)]


def test_parallel_run_writes_the_same_numbers(tmp_path):
    tasks = [small_task(j=j, steps=120, burn_in=40) for j in (0, 5, 10)]
    rs.run_tasks(tasks, str(tmp_path / 'a.csv'), workers=1, progress=False)
    rs.run_tasks(tasks, str(tmp_path / 'b.csv'), workers=2, progress=False)
    key = lambda r: r['j_index']
    a = sorted(rs.read_rows(str(tmp_path / 'a.csv')), key=key)
    b = sorted(rs.read_rows(str(tmp_path / 'b.csv')), key=key)
    assert [r['chi'] for r in a] == [r['chi'] for r in b]


def test_mixing_parameters_in_one_file_is_refused(tmp_path):
    path = str(tmp_path / 'sweep.csv')
    rs.run_tasks([small_task(steps=120, burn_in=40)], path, workers=1, progress=False)
    other = rs.make_task('short', C.BASE_SIZE, 8, 0, n0=144, steps=120, burn_in=40, debug=True)
    with pytest.raises(RuntimeError):
        rs.run_tasks([other], path, workers=1, progress=False)


def fake_rows(failed):
    """rows of a finished pass: 3 replications on a 4-point grid; `failed` = {(rep, j)}."""
    return [{'config': 'short', 'size_index': C.BASE_SIZE, 'j_index': j, 'rep': r,
             'failed_at': '7' if (r, j) in failed else ''} for r in range(3) for j in range(4)]


def test_top_up_replaces_a_replication_with_too_many_failed_points():
    rows = fake_rows({(1, 0), (1, 1), (1, 2), (1, 3)})
    tasks = rs.top_up_tasks(rows, ['short'], [C.BASE_SIZE], range(4), n_reps=3, n0=288, debug=True)
    assert sorted((t.rep, t.j_index) for t in tasks) == [(3, j) for j in range(4)]


def test_top_up_keeps_a_replication_with_three_failed_points_but_replaces_its_j_fix_run():
    rows = fake_rows({(1, 0), (1, 1), (1, 2)})        # 3 failed points: replication stays
    tasks = rs.top_up_tasks(rows, ['short'], [C.BASE_SIZE], range(4), n_reps=3, n0=288, j_fix_index=2, debug=True)
    assert [(t.rep, t.j_index) for t in tasks] == [(3, 2)]   # only H1 needs a new run at J_fix


def test_top_up_needs_nothing_when_all_runs_finished():
    tasks = rs.top_up_tasks(fake_rows(set()), ['short'], [C.BASE_SIZE], range(4), n_reps=3, n0=288,
                            j_fix_index=2, debug=True)
    assert tasks == []


# ---------------------------------------------------------------- fixed volume spread

def test_fixed_volume_spread_on_a_known_book():
    random.seed(0)
    exchange = ExchangeAgent(volume=10)
    exchange.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
    for price, qty in ((100.2, 5), (100.5, 10)):
        exchange.order_book['ask'].append(Order(price, qty, 'ask'))
    for price, qty in ((99.8, 5), (99.0, 10)):
        exchange.order_book['bid'].append(Order(price, qty, 'bid'))
    assert fixed_volume_spread(exchange, 5) == pytest.approx(100.2 - 99.8)        # the quoted spread
    buy = (5 * 100.2 + 5 * 100.5) / 10
    sell = (5 * 99.8 + 5 * 99.0) / 10
    assert fixed_volume_spread(exchange, 10) == pytest.approx(buy - sell)
    assert fixed_volume_spread(exchange, 16) != fixed_volume_spread(exchange, 16)  # NaN: not enough depth
