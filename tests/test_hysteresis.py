"""
Checks for experiments/run_hysteresis.py: loop area, one short loop up and down the grid, resume.
"""
import pytest

from experiments import config as C
from experiments import run_hysteresis as rh

def test_loop_area():
    assert rh.loop_area([0.1, 0.5, 0.9], [0.3, 0.9, 0.9], 0.1) == pytest.approx((0.2 + 0.4) * 0.1)


def test_hysteresis_run_goes_up_and_down_without_reset(tmp_path):
    grid = (0, 10, 20)
    task = rh.make_task('short', C.BASE_SIZE, 0, n0=288, grid=grid, burn_in=50, level_steps=60, discard=20,
                        debug=True)
    row = rh.simulate(task)
    assert row['failed_at'] == ''
    ups = [row[f'up_{C.J_GRID[j]:.1f}'] for j in grid]
    downs = [row[f'down_{C.J_GRID[j]:.1f}'] for j in grid]
    assert all(0 <= v <= 1 for v in ups + downs)
    assert row['area'] == pytest.approx(rh.loop_area(ups, downs, 1.0))
    path = str(tmp_path / 'h.csv')
    assert len(rh.run_tasks([task], path, workers=1, progress=False)) == 1
    assert len(rh.run_tasks([task], path, workers=1, progress=False)) == 0      # resume


def test_schedule_burn_in_then_up_then_down_without_reset(monkeypatch):
    """Spy on the simulator and on the averaging window: the protocol of prereg section 5."""
    from AgentBasedModel.simulator import Simulator
    from experiments import measures
    calls, windows = [], []
    original_simulate, original_m = Simulator.simulate, measures.magnetisation_signed

    def spy_simulate(self, n_iter, silent=False):
        chartists = [t for t in self.traders if t.type == 'Chartist']
        calls.append((len(self.info.prices), chartists[0].J, n_iter))
        return original_simulate(self, n_iter, silent)

    def spy_m(market, start=0, stop=None):
        windows.append((start, stop))
        return original_m(market, start, stop)

    monkeypatch.setattr(Simulator, 'simulate', spy_simulate)
    monkeypatch.setattr(rh, 'magnetisation_signed', spy_m)
    grid = (0, 10, 20)
    row = rh.simulate(rh.make_task('none', C.BASE_SIZE, 0, n0=0, grid=grid, burn_in=40, level_steps=30, discard=10,
                                   debug=True))
    assert row['failed_at'] == ''
    js = [C.J_GRID[j] for j in grid]
    expected_J = [0.0] + js + js[::-1]                       # burn-in at J = 0, up, then down (2.0 twice)
    assert [J for _, J, _ in calls] == expected_J
    assert [n for _, _, n in calls] == [40] + [30] * 6
    starts = [s for s, _, _ in calls]
    assert starts == [0] + [40 + 30 * k for k in range(6)]  # the state is never reset: steps keep counting
    assert windows == [(s + 10, s + 30) for s in starts[1:]]  # first 10 steps of each level dropped


def test_another_grid_or_protocol_in_the_same_file_is_refused(tmp_path):
    path = str(tmp_path / 'h.csv')
    t1 = rh.make_task('none', C.BASE_SIZE, 0, n0=0, grid=(0, 10), burn_in=20, level_steps=20, discard=5, debug=True)
    rh.run_tasks([t1], path, workers=1, progress=False)
    t2 = rh.make_task('short', C.BASE_SIZE, 0, n0=288, grid=(0, 10, 20), burn_in=20, level_steps=20, discard=5,
                      debug=True)
    with pytest.raises(RuntimeError):
        rh.run_tasks([t2], path, workers=1, progress=False)
    t3 = rh.make_task('short', C.BASE_SIZE, 0, n0=288, grid=(0, 10), burn_in=20, level_steps=25, discard=5, debug=True)
    with pytest.raises(RuntimeError):
        rh.run_tasks([t3], path, workers=1, progress=False)
