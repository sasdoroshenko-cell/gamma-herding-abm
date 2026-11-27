"""
Hysteresis loops for hypothesis H2 (task C-6; docs/prereg.md, section 5).

One long run per (configuration, replication): 500 steps of burn-in at J = 0, then J goes up the
grid 0 -> 2 and back down 2 -> 0, 200 steps on each level, without resetting the state between
levels. On each level the first 50 steps are dropped and m(J) is the mean |m| over the other 150.
Both directions visit J = 2, so there are 2 * 21 = 42 levels and 500 + 42 * 200 = 8900 steps.

The loop area A = sum_J |m_up(J) - m_down(J)| * dJ is written with the curves.
A failed run is replaced by the next seed in the stream (prereg, section 7).

Usage (from the repository root, with .venv active):
    python experiments/run_hysteresis.py --scale main      # short, none, long x 60 replications (N_REPS)
    python experiments/run_hysteresis.py --scale debug --reps 2 --grid-step 0.5 --level-steps 100

Output: results/summary/hysteresis.csv (one row per run: area and both curves).
"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AgentBasedModel.builder import build_market  # noqa: E402
from experiments import config as C  # noqa: E402
from experiments.measures import magnetisation_signed, failure_reason  # noqa: E402

KEY = ('config', 'size_index', 'rep')
PARAMS = ('book', 'seed', 'burn_in', 'level_steps', 'discard', 'grid', 'model')


@dataclass(frozen=True)
class HysteresisTask:
    config: str
    size_index: int
    rep: int
    book: float
    seed: int
    grid: tuple            # indices into config.J_GRID, increasing
    burn_in: int = C.HYST_BURN_IN
    level_steps: int = C.HYST_STEP
    discard: int = C.HYST_DISCARD

    @property
    def key(self):
        return self.config, self.size_index, self.rep

    @property
    def model(self):
        return C.model_id(C.market_config(self.config, 0.0, self.book))


def make_task(config, size_index, rep, n0, grid=None, burn_in=C.HYST_BURN_IN, level_steps=C.HYST_STEP,
              discard=C.HYST_DISCARD, debug=False):
    grid = tuple(range(len(C.J_GRID))) if grid is None else tuple(grid)
    return HysteresisTask(config=config, size_index=size_index, rep=rep, book=C.book_size(config, size_index, n0),
                          seed=C.seed(config, size_index, C.HYST_J_INDEX, rep, debug=debug), grid=grid,
                          burn_in=burn_in, level_steps=level_steps, discard=discard)


def curve_columns(grid):
    return ([f'up_{C.J_GRID[j]:.1f}' for j in grid] + [f'down_{C.J_GRID[j]:.1f}' for j in grid])


def loop_area(m_up, m_down, dJ) -> float:
    """A = sum_J |m_up(J) - m_down(J)| * dJ over the points of the grid (NaN if a point is missing)."""
    m_up, m_down = np.asarray(m_up, dtype=float), np.asarray(m_down, dtype=float)
    return float(np.sum(np.abs(m_up - m_down)) * dJ)


def simulate(task: HysteresisTask) -> dict:
    t0 = time.perf_counter()
    js = [C.J_GRID[j] for j in task.grid]
    levels = [('up', J) for J in js] + [('down', J) for J in reversed(js)]
    market = build_market(C.market_config(task.config, 0.0, task.book), seed=task.seed)
    sim = market.simulator
    curves = {}
    failed_at, error = None, ''
    try:
        sim.simulate(task.burn_in, silent=True)
        for direction, J in levels:
            for ch in market.chartists:
                ch.J = J
            start = len(sim.info.prices)
            sim.simulate(task.level_steps, silent=True)
            m = np.abs(magnetisation_signed(market, start + task.discard, start + task.level_steps))
            curves[f'{direction}_{J:.1f}'] = float(m.mean())
    except Exception as exc:
        error = failure_reason(exc)
        failed_at = len(sim.info.prices)

    row = {'config': task.config, 'size_index': task.size_index, 'rep': task.rep, 'book': task.book,
           'seed': task.seed, 'burn_in': task.burn_in, 'level_steps': task.level_steps, 'discard': task.discard,
           'grid': ' '.join(str(j) for j in task.grid), 'model': task.model,
           'failed_at': '' if failed_at is None else failed_at, 'error': error}
    cols = curve_columns(task.grid)
    if failed_at is None:
        dJ = js[1] - js[0] if len(js) > 1 else 0.0
        row['area'] = loop_area([curves[c] for c in cols[:len(js)]], [curves[c] for c in cols[len(js):]], dJ)
    else:
        row['area'] = math.nan
    for c in cols:
        row[c] = curves.get(c, math.nan)
    row['runtime_s'] = round(time.perf_counter() - t0, 3)
    return row


def fields(grid):
    return list(KEY) + ['book', 'seed', 'burn_in', 'level_steps', 'discard', 'grid', 'model', 'failed_at', 'error',
                        'runtime_s', 'area'] + curve_columns(grid)


def read_rows(path) -> list:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('size_index', 'rep', 'seed', 'burn_in', 'level_steps', 'discard'):
            r[k] = int(r[k])
        r['book'] = float(r['book'])
    return rows


def run_tasks(tasks, path, workers=None, progress=True) -> list:
    """Run the tasks that are not yet in `path`; each row is appended as soon as it is ready."""
    if not tasks:
        return []
    grid = tasks[0].grid
    if any(t.grid != grid for t in tasks):
        raise ValueError('all tasks of one file must use the same grid')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline='') as f:
            header = next(csv.reader(f))
        if header != fields(grid):
            raise RuntimeError(f"{path} was written with other columns or another grid. "
                               f"Use another --out or delete the file.")
    rows = read_rows(path)
    # every loop of a file shares the protocol and the model, so the curves and areas are comparable
    file_level = {(r['burn_in'], r['level_steps'], r['discard'], r['grid'], r['model']) for r in rows}
    for t in tasks:
        mine = (t.burn_in, t.level_steps, t.discard, ' '.join(map(str, t.grid)), t.model)
        if file_level and file_level != {mine}:
            raise RuntimeError(f"{t.key}: protocol or model differ from the loops in {path}. Use another --out.")
    done = {}
    for r in rows:
        done[(r['config'], r['size_index'], r['rep'])] = r
    for t in tasks:
        r = done.get(t.key)
        if r is not None and (abs(r['book'] - t.book) > 1e-9 or r['seed'] != t.seed):
            raise RuntimeError(f"{t.key}: parameters differ from the file {path}. Use another --out.")
    todo = [t for t in dict.fromkeys(tasks) if t.key not in done]
    if not todo:
        return []
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new_file = not rows and (not os.path.exists(path) or os.path.getsize(path) == 0)
    workers = workers or os.cpu_count()
    out = []
    t0 = time.perf_counter()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields(grid))
        if new_file:
            writer.writeheader()
        pool = Pool(min(workers, len(todo))) if workers > 1 else None
        iterator = pool.imap_unordered(simulate, todo, chunksize=1) if pool else map(simulate, todo)
        try:
            for i, row in enumerate(iterator, 1):
                writer.writerow(row)
                f.flush()
                out.append(row)
                if progress:
                    print(f"  {i}/{len(todo)} петель, {time.perf_counter() - t0:.0f} с", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return out


def hysteresis(configs, n_reps, n0, path, workers=None, size_index=C.BASE_SIZE, grid=None, burn_in=C.HYST_BURN_IN,
               level_steps=C.HYST_STEP, discard=C.HYST_DISCARD, debug=False, max_rounds=5):
    """n_reps finished loops per configuration; failed runs are replaced by the next seed."""
    def task(config, rep):
        return make_task(config, size_index, rep, n0, grid, burn_in, level_steps, discard, debug)

    tasks = [task(c, rep) for c in configs for rep in range(n_reps)]
    print(f"{len(tasks)} петель в плане, файл {os.path.relpath(path, ROOT)}")
    run_tasks(tasks, path, workers)
    for _ in range(max_rounds):
        rows = read_rows(path)
        extra = []
        for c in configs:
            mine = {r['rep']: r for r in rows if r['config'] == c and r['size_index'] == size_index}
            ok = sum(r['failed_at'] == '' for r in mine.values())
            rep = n_reps
            while ok < n_reps:
                if rep > C.MAX_REP:
                    print(f"  ВНИМАНИЕ: {c}: поток сидов исчерпан, готовых петель {ok} из {n_reps}")
                    break
                if rep not in mine:
                    extra.append(task(c, rep))
                    ok += 1
                rep += 1
        if not extra:
            break
        print(f"замены прерванных петель: {len(extra)}")
        run_tasks(extra, path, workers)
    return read_rows(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scale', choices=['main', 'debug'], required=True)
    parser.add_argument('--configs', nargs='+', choices=C.CONFIGS, default=list(C.CONFIGS))
    parser.add_argument('--reps', type=int, help='debug only')
    parser.add_argument('--grid-step', type=float, default=0.1, help='debug only: multiple of 0.1')
    parser.add_argument('--level-steps', type=int, help='debug only')
    parser.add_argument('--book', type=float, help=f'debug only: n0 (default {C.N0_PROVISIONAL})')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    if args.scale == 'debug':
        n0 = args.book if args.book is not None else C.N0_PROVISIONAL
        grid = list(range(0, len(C.J_GRID), round(args.grid_step / C.J_STEP)))
        level_steps = args.level_steps or C.HYST_STEP
        discard = min(C.HYST_DISCARD, level_steps // 4)
        n_reps, debug = args.reps or 2, True
        path = args.out or os.path.join(ROOT, 'results', 'debug', 'hysteresis.csv')
    else:
        if any(v is not None for v in (args.reps, args.level_steps, args.book)) or args.grid_step != 0.1:
            parser.error('--reps, --grid-step, --level-steps and --book are for --scale debug only')
        C.require_frozen()
        n0, grid, level_steps, discard, n_reps, debug = C.N0, None, C.HYST_STEP, C.HYST_DISCARD, C.N_REPS, False
        path = args.out or os.path.join(ROOT, 'results', 'summary', 'hysteresis.csv')

    t0 = time.perf_counter()
    rows = hysteresis(args.configs, n_reps, n0, path, args.workers, grid=grid, level_steps=level_steps,
                      discard=discard, debug=debug)
    print(f"готово за {time.perf_counter() - t0:.0f} с")
    for c in args.configs:
        mine = [r for r in rows if r['config'] == c]
        areas = [float(r['area']) for r in mine if r['failed_at'] == '']
        print(f"  {c:5s}: петель {len(mine)}, прервано {sum(r['failed_at'] != '' for r in mine)}, "
              f"медиана площади {np.median(areas) if areas else float('nan'):.3f}")


if __name__ == '__main__':
    main()
