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
