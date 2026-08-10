"""Which moment basis pins down the absorbance profile more tightly?

For a given basis and a given true measure, the moments are a handful of real
numbers. Many different measures share them. The question that matters for
rendering is how much that ambiguity costs: across every non-negative measure
consistent with the moments, how far apart can F(z0) -- the fraction of the mass
in front of z0 -- possibly be?

That is a linear program. Minimising it gives the tightest achievable lower
bound for that basis, which is exactly what an MBOIT-style reconstruction
returns. Maximising gives the tightest upper bound. The gap between them is the
information the basis genuinely fails to carry, independent of any particular
reconstruction algorithm.

Budget is held equal: a complex trigonometric moment costs two floats, so two
complex moments sit against four power moments.
"""
import numpy as np
from scipy.optimize import linprog

GRID = np.linspace(0.0, 1.0, 1201)


def power_basis(order=4):
    return [(f"z^{k}", lambda z, k=k: z ** k) for k in range(1, order + 1)]


def trig_basis(harmonics=2, omega=np.pi):
    rows = []
    for k in range(1, harmonics + 1):
        rows.append((f"cos{k}", lambda z, k=k: np.cos(k * omega * z)))
        rows.append((f"sin{k}", lambda z, k=k: np.sin(k * omega * z)))
    return rows


def slab(low, high):
    d = ((GRID >= low) & (GRID <= high)).astype(float)
    return d / d.sum()


def diracs(*positions):
    d = np.zeros_like(GRID)
    for p in positions:
        d[np.argmin(np.abs(GRID - p))] = 1.0
    return d / d.sum()


def true_cdf(density, z0):
    return density[GRID < z0].sum()


def bounds_at(density, basis, z0):
    """Tightest lower and upper bound on F(z0) from these moments."""
    rows = [np.ones_like(GRID)] + [f(GRID) for _, f in basis]
    target = [1.0] + [float(np.dot(f(GRID), density)) for _, f in basis]
    objective = (GRID < z0).astype(float)

    out = []
    for sign in (1.0, -1.0):
        r = linprog(sign * objective, A_eq=np.array(rows), b_eq=np.array(target),
                    bounds=(0.0, None), method="highs")
        out.append(sign * r.fun if r.success else np.nan)
    return out[0], out[1]


BASES = [
    ("power x4      ", power_basis(4)),
    ("trig x2 (w=pi)", trig_basis(2, np.pi)),
    ("trig x2 (w=2pi)", trig_basis(2, 2 * np.pi)),
]

CASES = [
    ("uniform slab [0.4,0.6]", slab(0.4, 0.6)),
    ("uniform slab [0.3,0.8]", slab(0.3, 0.8)),
    ("uniform slab [0.05,0.95]", slab(0.05, 0.95)),
    ("two surfaces {0.3,0.7}", diracs(0.3, 0.7)),
]

for name, density in CASES:
    print(f"\n=== {name} ===")
    print(f"{'z0':>5} {'true':>6} | " + " | ".join(f"{b:>22}" for b, _ in BASES))
    print(f"{'':>5} {'':>6} | " + " | ".join(f"{'lower  upper   gap':>22}" for _ in BASES))
    widths = {b: [] for b, _ in BASES}
    lower_err = {b: [] for b, _ in BASES}
    for z0 in [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]:
        truth = true_cdf(density, z0)
        cells = []
        for bname, basis in BASES:
            lo, hi = bounds_at(density, basis, z0)
            widths[bname].append(hi - lo)
            lower_err[bname].append(truth - lo)
            cells.append(f"{lo:6.3f} {hi:6.3f} {hi - lo:7.3f}")
        print(f"{z0:>5.2f} {truth:>6.3f} | " + " | ".join(cells))
    print("  mean gap:      " + "  ".join(
        f"{b.strip()}={np.mean(widths[b]):.3f}" for b, _ in BASES))
    print("  mean lower err:" + "  ".join(
        f"{b.strip()}={np.mean(lower_err[b]):.3f}" for b, _ in BASES))
