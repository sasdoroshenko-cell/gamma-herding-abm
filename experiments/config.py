"""
All parameters of the project runs in one place (task C-3; docs/prereg.md, sections 3, 4 and 7).

Every run is fixed by four numbers: configuration, book size, point of the J grid and replication.
The seed is a function of these four numbers only (prereg, section 4), so any run can be repeated
on any machine, and the runs can be split between laptops and merged afterwards.

Three values are computed after calibration by the rules of the prereg and are still unset:
N0 (book size), J_FIX (hypothesis H1) and J_REF (calibration and lambda). Full-scale runs refuse to
start while they are None; debug and pilot runs use the *_PROVISIONAL values and a separate seed
stream, so they never coincide with the runs that go into the report.
"""
from dataclasses import replace

from AgentBasedModel.builder import MarketConfig

# ---------------------------------------------------------------- model (prereg, section 2)
# Parameters fixed by the prereg. Calibration (task B-7) adds its parameters here before config-v1.
BASE = MarketConfig(a1=0.6, a2=1.0, v1=0.1)

# ---------------------------------------------------------------- configurations (prereg, section 3)
CONFIGS = ('short', 'none', 'long')
# Number of a configuration in the seed formula; also its rank for Kendall's tau (short = 1, none = 2, long = 3)
CONFIG_INDEX = {'short': 1, 'none': 2, 'long': 3}

SIZE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)   # book sizes for the sensitivity analysis, in units of n0
BASE_SIZE = SIZE_FACTORS.index(1.0)         # size index of the main experiment (n = n0)

N0 = None              # straddles in the whole book; Frey rule (prereg, section 3), a multiple of 12
N0_PROVISIONAL = 288   # debug value from scripts/mechanism_check.py; never used in full runs

# ---------------------------------------------------------------- protocol (prereg, section 4)
J_GRID = tuple(round(0.1 * k, 1) for k in range(21))   # 0.0, 0.1, ..., 2.0
J_STEP = 0.1
STEPS = 1500
BURN_IN = 500            # statistics use steps 501-1500, i.e. indices 500..1499 of the recorded series
N_REPS = 30
MAX_FAILED_POINTS = 3    # more failed grid points in a replication -> it is replaced (prereg, section 7)
MAX_REP = 99             # the seed formula leaves room for replication numbers 0..99

J_FIX = None             # smallest grid point with mean |m| >= 0.8 without dealers (prereg, section 4)
J_FIX_PROVISIONAL = 1.1  # value from the uncalibrated debug sweep (docs/debug_sweep_J.png)
J_REF = None             # largest grid point with mean |m| <= 0.3 without dealers
M_FIX_LEVEL = 0.8
M_REF_LEVEL = 0.3

# ---------------------------------------------------------------- hysteresis, hypothesis H2 (prereg, section 5)
HYST_BURN_IN = 500       # steps at J = 0 before the first level
HYST_STEP = 200          # steps on each level of J
HYST_DISCARD = 50        # first steps of a level that are not averaged
HYST_J_INDEX = 99        # seed slot of the hysteresis runs: not a point of the grid, so no overlap with sweeps

# ---------------------------------------------------------------- measurements written for every run
FVS_VOLUMES = (10, 20, 50)   # Q for the fixed volume spread (lambda estimate, prereg section 3)
FVS_EVERY = 10               # the book is measured every 10 steps after burn-in

# ---------------------------------------------------------------- analysis (prereg, section 6)
ALPHA = 0.05
N_PERM = 10_000
N_BOOT = 10_000
CI_LEVEL = 0.95
ANALYSIS_SEED = 20261122     # random generator of permutations and bootstrap; fixed so numbers repeat

# ---------------------------------------------------------------- seeds
DEBUG_SEED_OFFSET = 10_000_000   # debug and pilot runs: outside the range of the full-run seeds


def seed(config: str, size_index: int, j_index: int, rep: int, debug: bool = False) -> int:
    """
    Seed of one simulation (prereg, section 4):
        1 000 000 * (configuration) + 10 000 * (book size) + 100 * (point of the J grid) + (replication)
    Configurations are numbered short = 1, none = 2, long = 3; book sizes and grid points from 0;
    replications from 0 (0..29 are the main ones, 30 and later replace failed ones).
    Hysteresis runs use j_index = HYST_J_INDEX.
    """
    if config not in CONFIG_INDEX:
        raise ValueError(f"unknown configuration {config!r}")
    if not 0 <= size_index < 100:
        raise ValueError("size_index must be in 0..99")
    if not 0 <= j_index <= 99:
        raise ValueError("j_index must be in 0..99")
    if not 0 <= rep <= MAX_REP:
        raise ValueError(f"rep must be in 0..{MAX_REP}")
    s = 1_000_000 * CONFIG_INDEX[config] + 10_000 * size_index + 100 * j_index + rep
    return s + DEBUG_SEED_OFFSET if debug else s


def j_index(J: float) -> int:
    """Position of J in J_GRID."""
    for i, g in enumerate(J_GRID):
        if abs(g - J) < 1e-9:
            return i
    raise ValueError(f"J = {J} is not a point of the grid")


def book_size(config: str, size_index: int, n0: float) -> float:
    """Straddles in the whole book: SIZE_FACTORS[size_index] * n0, and 0 for 'none'."""
    if config == 'none':
        return 0.0
    return SIZE_FACTORS[size_index] * n0


def market_config(config: str, J: float, book: float) -> MarketConfig:
    """MarketConfig of one run: BASE with the coupling J and the dealers' book."""
    return replace(BASE, J=J, dealer_side=config, dealer_book=book if config != 'none' else 0.0)


def require_frozen():
    """Full runs are allowed only after N0 and J_FIX are written here (prereg, sections 3-4)."""
    missing = [name for name, value in (('N0', N0), ('J_FIX', J_FIX)) if value is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not set in experiments/config.py: full runs start only after calibration "
            f"and the rules of docs/prereg.md (sections 3-4). For a test use --scale debug.")
