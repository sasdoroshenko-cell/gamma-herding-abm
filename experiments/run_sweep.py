"""
Sweeps over J (task C-4; docs/prereg.md, sections 4 and 7).

One task = one simulation = (configuration, book size, point of the J grid, replication).
Each finished run is appended to a CSV at once, so a crash loses at most the runs in progress;
a second start reads the CSV and skips everything already there.

Failed runs (e.g. one side of the book emptied) are written with the step of failure and are
never dropped. After the main pass the runner adds replacement replications by the rules of the
prereg (section 7): a replication with more than 3 failed grid points is replaced by the next one
in the seed stream, and a failed run at J_fix is replaced by the next seed for hypothesis H1.

Usage (from the repository root, with .venv active):
    python experiments/run_sweep.py --scale main          # 3 configurations x 21 J x 30 replications
    python experiments/run_sweep.py --scale sensitivity   # + the other four book sizes
    python experiments/run_sweep.py --scale debug --reps 2 --grid-step 0.5 --steps 600 --burn-in 200
    python experiments/run_sweep.py --scale main --configs short       # split the work between laptops

Output: results/summary/sweep.csv for main and sensitivity (one file: the main experiment is the
book size n0 of the sensitivity analysis), results/debug/sweep.csv for debug runs.
"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AgentBasedModel.builder import build_market  # noqa: E402
from experiments import config as C  # noqa: E402
from experiments.measures import fixed_volume_spread, run_summary, magnetisation_signed, failure_reason  # noqa: E402

KEY = ('config', 'size_index', 'j_index', 'rep')
PARAMS = ('J', 'book', 'steps', 'burn_in', 'seed', 'overrides', 'model')
FIELDS = (KEY + PARAMS + ('failed_at', 'error', 'runtime_s',
          'mean_abs_m', 'var_abs_m', 'chi', 'mean_m', 'sigma', f'sigma_{C.SIGMA_HORIZON}', 'acf1', 'kurtosis',
          'mean_abs_return', 'spread', 'kappa')
          + tuple(f'fvs_{q}' for q in C.FVS_VOLUMES)
          + ('dealer_share', 'dealer_flow_abs', 'dealer_gamma', 'corr_flow_past_move'))


def format_overrides(values: dict) -> str:
    """MarketConfig fields that differ from config.BASE, e.g. {'a2': 30} -> 'a2=30'."""
    return ';'.join(f'{k}={v:g}' for k, v in sorted(values.items()))


def parse_overrides(text: str) -> dict:
    return {k: float(v) for k, v in (item.split('=') for item in text.split(';') if item)}


@dataclass(frozen=True)
class Task:
    config: str
    size_index: int
    j_index: int
    rep: int
    J: float
    book: float
    steps: int
    burn_in: int
    seed: int
    series_path: str = ''   # if set, the full time series of the run are saved there (.npz)
    overrides: str = ''     # MarketConfig fields that differ from config.BASE (power checks only)

    @property
    def key(self):
        return self.config, self.size_index, self.j_index, self.rep

    def market_config(self):
        return replace(C.market_config(self.config, self.J, self.book), **parse_overrides(self.overrides))

    @property
    def model(self):
        return C.model_id(self.market_config())


def make_task(config, size_index, j_index, rep, n0, steps=C.STEPS, burn_in=C.BURN_IN, debug=False,
              series_path='', overrides=''):
    return Task(config=config, size_index=size_index, j_index=j_index, rep=rep, J=C.J_GRID[j_index],
                book=C.book_size(config, size_index, n0), steps=steps, burn_in=burn_in,
                seed=C.seed(config, size_index, j_index, rep, debug=debug), series_path=series_path,
                overrides=overrides)


def simulate(task: Task) -> dict:
    """Run one simulation and return its row for the CSV."""
    t0 = time.perf_counter()
    market = build_market(task.market_config(), seed=task.seed)
    sim = market.simulator
    fvs = {q: [] for q in C.FVS_VOLUMES}
    failed_at, error = None, ''
    try:
        sim.simulate(task.burn_in, silent=True)
        done = task.burn_in
        while done < task.steps:   # measure the book every FVS_EVERY steps; the simulation is the same
            for q in C.FVS_VOLUMES:
                fvs[q].append(fixed_volume_spread(sim.exchange, q))
            chunk = min(C.FVS_EVERY, task.steps - done)
            sim.simulate(chunk, silent=True)
            done += chunk
    except Exception as exc:
        error = failure_reason(exc)          # a bug in the code is raised here, not counted as a failed run
        failed_at = len(sim.info.prices)     # step at which the run stopped

    row = {k: getattr(task, k) for k in KEY + PARAMS}
    row['failed_at'] = '' if failed_at is None else failed_at
    row['error'] = error
    if failed_at is None:
        row.update(run_summary(market, task.burn_in, task.steps, fvs, C.SIGMA_HORIZON))
    if task.series_path:
        save_series(market, task.series_path)
    row['runtime_s'] = round(time.perf_counter() - t0, 3)
    return {f: row.get(f, math.nan) for f in FIELDS}


def save_series(market, path):
    """Full series of one run for figures: price, m, dealer flow and book gamma per step."""
    import numpy as np
    from AgentBasedModel.builder import dealer_gamma, dealer_flow
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gamma = [math.nan if g is None else g for g in dealer_gamma(market)]
    np.savez_compressed(path, price=np.asarray(market.simulator.info.prices, dtype=float),
                        m=magnetisation_signed(market), dealer_flow=np.asarray(dealer_flow(market), dtype=float),
                        dealer_gamma=np.asarray(gamma, dtype=float))


# ---------------------------------------------------------------- CSV with resume

def read_rows(path) -> list:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('size_index', 'j_index', 'rep', 'steps', 'burn_in', 'seed'):
            r[k] = int(r[k])
        r['J'], r['book'] = float(r['J']), float(r['book'])
    return rows


def check_compatible(rows, tasks):
    """
    Refuse to mix runs with different parameters in one file: a task whose key is already in the
    file must have the same parameters, and every run of a file must share the run length and
    the model (e.g. not a provisional and a calibrated model, or two values of a2).
    """
    file_level = {(r['steps'], r['burn_in'], r['model']) for r in rows}
    if len(file_level) > 1:
        raise RuntimeError(f"the file already mixes run lengths or models: {sorted(file_level)}")
    for t in tasks:
        if file_level and (t.steps, t.burn_in, t.model) not in file_level:
            raise RuntimeError(f"{t.key}: model {t.model} / {t.steps} steps differ from the runs in the file "
                               f"{sorted(file_level)}. Use another --out or delete the file.")
    done = {tuple(r[k] for k in KEY): r for r in rows}
    for t in tasks:
        r = done.get(t.key)
        if r is None:
            continue
        for p in PARAMS:
            old, new = r[p], getattr(t, p)
            if (abs(old - new) > 1e-9) if isinstance(new, float) else old != new:
                raise RuntimeError(f"{t.key}: {p} = {old} in the file, but {new} now. "
                                   f"Use another --out or delete the file.")


def run_tasks(tasks, path, workers=None, progress=True) -> list:
    """Run the tasks that are not yet in `path`, appending each row as soon as it is ready."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline='') as f:
            header = next(csv.reader(f))
        if header != list(FIELDS):
            raise RuntimeError(f"{path} was written with other columns (an older version of the runner). "
                               f"Use another --out or delete the file.")
    rows = read_rows(path)
    check_compatible(rows, tasks)
    done = {tuple(r[k] for k in KEY) for r in rows}
    todo = [t for t in dict.fromkeys(tasks) if t.key not in done]
    if not todo:
        return []
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    workers = workers or os.cpu_count()
    results = []
    t0 = time.perf_counter()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        if workers == 1:
            iterator = map(simulate, todo)
            pool = None
        else:
            pool = Pool(workers)
            iterator = pool.imap_unordered(simulate, todo, chunksize=1)
        try:
            for i, row in enumerate(iterator, 1):
                writer.writerow(row)
                f.flush()
                results.append(row)
                if progress and (i % max(1, len(todo) // 20) == 0 or i == len(todo)):
                    elapsed = time.perf_counter() - t0
                    print(f"  {i}/{len(todo)} прогонов, {elapsed:.0f} с, осталось ~{elapsed / i * (len(todo) - i):.0f} с",
                          flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return results


# ---------------------------------------------------------------- which runs are needed

def replication_status(rows, config, size_index, grid_indices, max_failed=C.MAX_FAILED_POINTS):
    """
    For every replication of (config, size): 'valid' (all grid points run, at most max_failed failed),
    'replaced' (too many failed points) or 'incomplete' (not all points run yet).
    """
    by_rep = {}
    for r in rows:
        if r['config'] == config and r['size_index'] == size_index and r['j_index'] in grid_indices:
            by_rep.setdefault(r['rep'], []).append(r)
    status = {}
    for rep, rs in sorted(by_rep.items()):
        if {r['j_index'] for r in rs} != set(grid_indices):
            status[rep] = 'incomplete'
        elif sum(r['failed_at'] != '' for r in rs) > max_failed:
            status[rep] = 'replaced'
        else:
            status[rep] = 'valid'
    return status


def top_up_tasks(rows, configs, sizes, grid_indices, n_reps, n0, j_fix_index=None, steps=C.STEPS,
                 burn_in=C.BURN_IN, debug=False):
    """Replacement runs required by the prereg (section 7) given the rows already in the file."""
    tasks = []
    for config in configs:
        for size in sizes:
            status = replication_status(rows, config, size, grid_indices)
            n_valid = sum(s == 'valid' for s in status.values())
            rep = n_reps
            while n_valid < n_reps:   # main hypothesis: whole replications
                if rep > C.MAX_REP:
                    print(f"  ВНИМАНИЕ: {config}, размер {size}: поток сидов исчерпан, валидных репликаций "
                          f"{n_valid} из {n_reps}")
                    break
                if status.get(rep) != 'valid' and status.get(rep) != 'replaced':
                    tasks += [make_task(config, size, j, rep, n0, steps, burn_in, debug) for j in grid_indices]
                    n_valid += 1   # optimistic: if it fails again, the next pass adds another one
                rep += 1
            if j_fix_index is not None and size == C.BASE_SIZE:   # H1: n_reps finished runs at J_fix
                existing = {r['rep']: r for r in rows if r['config'] == config and r['size_index'] == size
                            and r['j_index'] == j_fix_index}
                scheduled = {t.rep for t in tasks
                             if t.config == config and t.size_index == size and t.j_index == j_fix_index}
                ok = sum(r['failed_at'] == '' for r in existing.values()) + len(scheduled - existing.keys())
                rep = n_reps
                while ok < n_reps:
                    if rep > C.MAX_REP:
                        print(f"  ВНИМАНИЕ: {config}: поток сидов при J_fix исчерпан")
                        break
                    if rep not in existing and rep not in scheduled:
                        tasks.append(make_task(config, size, j_fix_index, rep, n0, steps, burn_in, debug))
                        ok += 1
                    rep += 1
    return list(dict.fromkeys(tasks))


def sweep(configs, sizes, grid_indices, n_reps, n0, path, workers=None, steps=C.STEPS, burn_in=C.BURN_IN,
          debug=False, j_fix_index=None, series_reps=0, series_dir=None, top_up=True, max_rounds=5):
    """Main pass and up to `max_rounds` passes of replacement runs. Returns all rows of the file."""
    def series_path(config, size, j, rep):
        if rep < series_reps and size == C.BASE_SIZE and series_dir:
            return os.path.join(series_dir, f"{config}_s{size}_j{j:02d}_r{rep:02d}.npz")
        return ''

    tasks = [make_task(c, s, j, rep, n0, steps, burn_in, debug, series_path(c, s, j, rep))
             for c in configs for s in sizes for rep in range(n_reps) for j in grid_indices]
    print(f"{len(tasks)} прогонов в плане, файл {os.path.relpath(path, ROOT)}")
    run_tasks(tasks, path, workers)
    for _ in range(max_rounds if top_up else 0):
        extra = top_up_tasks(read_rows(path), configs, sizes, grid_indices, n_reps, n0, j_fix_index,
                             steps, burn_in, debug)
        if not extra:
            break
        print(f"замены по правилам prereg: {len(extra)} прогонов")
        run_tasks(extra, path, workers)
    rows = read_rows(path)
    for c in configs:
        for s in sizes:
            status = replication_status(rows, c, s, grid_indices)
            n_valid = sum(v == 'valid' for v in status.values())
            if n_valid < n_reps:
                print(f"  ВНИМАНИЕ: {c}, размер {s}: валидных репликаций {n_valid} из {n_reps} "
                      f"после замен — это указывается в отчёте")
    return rows


def report_failures(rows):
    """Failed runs per configuration and book size: the numbers the report has to state."""
    from collections import Counter
    total, failed = Counter(), Counter()
    for r in rows:
        total[(r['config'], r['size_index'])] += 1
        failed[(r['config'], r['size_index'])] += r['failed_at'] != ''
    for key in sorted(total):
        print(f"  {key[0]:5s} размер {key[1]}: прервано {failed[key]} из {total[key]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scale', choices=['main', 'sensitivity', 'debug'], required=True)
    parser.add_argument('--configs', nargs='+', choices=C.CONFIGS, default=list(C.CONFIGS))
    parser.add_argument('--reps', type=int, help='debug only: replications per point')
    parser.add_argument('--grid-step', type=float, default=0.1, help='debug only: multiple of 0.1')
    parser.add_argument('--steps', type=int, help='debug only')
    parser.add_argument('--burn-in', type=int, help='debug only')
    parser.add_argument('--book', type=float, help=f'debug only: n0 (default {C.N0_PROVISIONAL})')
    parser.add_argument('--series-reps', type=int, default=0, help='save full series for replications < k')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    if args.scale == 'debug':
        n0 = args.book if args.book is not None else C.N0_PROVISIONAL
        stride = round(args.grid_step / C.J_STEP)
        grid = list(range(0, len(C.J_GRID), stride))
        sizes, n_reps = [C.BASE_SIZE], args.reps or 2
        steps, burn_in = args.steps or C.STEPS, args.burn_in or C.BURN_IN
        path = args.out or os.path.join(ROOT, 'results', 'debug', 'sweep.csv')
        j_fix_index, debug = None, True
    else:
        if any(v is not None for v in (args.reps, args.steps, args.burn_in, args.book)) or args.grid_step != 0.1:
            parser.error('--reps, --grid-step, --steps, --burn-in and --book are for --scale debug only')
        C.require_frozen()
        n0, grid, n_reps, steps, burn_in = C.N0, list(range(len(C.J_GRID))), C.N_REPS, C.STEPS, C.BURN_IN
        sizes = [C.BASE_SIZE] if args.scale == 'main' else list(range(len(C.SIZE_FACTORS)))
        path = args.out or os.path.join(ROOT, 'results', 'summary', 'sweep.csv')
        j_fix_index, debug = C.j_index(C.J_FIX), False

    t0 = time.perf_counter()
    rows = sweep(args.configs, sizes, grid, n_reps, n0, path, args.workers, steps, burn_in, debug, j_fix_index,
                 args.series_reps, os.path.join(ROOT, 'results', 'series'))
    print(f"готово за {time.perf_counter() - t0:.0f} с; прерванные прогоны:")
    report_failures([r for r in rows if r['config'] in args.configs and r['size_index'] in sizes])


if __name__ == '__main__':
    main()
