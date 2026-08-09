"""Analytically generated sample datasets.

Everything here is derived from a formula rather than recorded from a file, so
the expected result is always known. That matters for judging a renderer: noise
looks the same whether it is drawn correctly or not, whereas a torus with the
wrong winding, a beam with the stress sign flipped, or a benzene ring that is
not planar are all obvious on sight.

Each generator returns a ``{buffer_name: ndarray}`` mapping ready to hand to
:meth:`iris3d.Client.upload_object`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "benzene",
    "cantilever_beam",
    "examples",
    "random_cloud",
    "torus_mesh",
    "torus_points",
]


def torus_points(
    n_major: int = 128,
    n_minor: int = 48,
    major_radius: float = 3.0,
    minor_radius: float = 1.0,
) -> dict[str, np.ndarray]:
    """A point cloud on a torus, with scalar and vector fields.

    Recognisable at a glance, and the fields are position-derived so a wrong
    colour mapping or a mis-scaled glyph is obvious.
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()

    ring = major_radius + minor_radius * np.cos(vv)
    positions = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), minor_radius * np.sin(vv)], axis=1
    )

    # Tangent along the major circle — should everywhere follow the ring.
    tangent = np.stack([-np.sin(uu), np.cos(uu), np.zeros_like(uu)], axis=1)

    return {
        "positions": positions.astype(np.float32),
        "height": positions[:, 2].astype(np.float32),
        "angle": uu.astype(np.float32),
        "tangent": tangent.astype(np.float32),
    }


def torus_mesh(
    n_major: int = 96,
    n_minor: int = 32,
    major_radius: float = 3.0,
    minor_radius: float = 1.0,
) -> dict[str, np.ndarray]:
    """A closed triangulated torus with analytic normals.

    A torus rather than a sphere on purpose: it has no poles, so the grid wraps
    cleanly in both directions and there are no degenerate triangles to muddy a
    winding or normals bug.
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()

    ring = major_radius + minor_radius * np.cos(vv)
    positions = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), minor_radius * np.sin(vv)], axis=1
    )
    normals = np.stack(
        [np.cos(vv) * np.cos(uu), np.cos(vv) * np.sin(uu), np.sin(vv)], axis=1
    )

    iu = np.arange(n_major)
    iv = np.arange(n_minor)
    iu, iv = np.meshgrid(iu, iv, indexing="ij")
    iu, iv = iu.ravel(), iv.ravel()

    def index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a % n_major) * n_minor + (b % n_minor)

    i00 = index(iu, iv)
    i10 = index(iu + 1, iv)
    i11 = index(iu + 1, iv + 1)
    i01 = index(iu, iv + 1)
    # Consistent winding across both triangles of each quad.
    indices = np.concatenate(
        [np.stack([i00, i10, i11], axis=1), np.stack([i00, i11, i01], axis=1)]
    )

    return {
        "positions": positions.astype(np.float32),
        "indices": indices.astype(np.uint32),
        "normals": normals.astype(np.float32),
    }


def cantilever_beam(
    nx: int = 60,
    ny: int = 16,
    nz: int = 8,
    length: float = 6.0,
    height: float = 1.0,
    width: float = 0.6,
    load: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Sample points through a cantilever with its analytic bending stress.

    Fixed at ``x = 0``, point load at the free end. Euler-Bernoulli, so:

    - ``sigma_xx = M(x) * y / I`` with ``M(x) = P (L - x)`` — largest at the
      fixed end, zero at the tip, and changing sign across the neutral axis.
    - parabolic transverse shear ``tau_xy``, zero at the top and bottom faces.

    Those three properties make a wrong tensor layout or a bad colour range
    immediately visible. The tensor is written in the six-component symmetric
    Voigt order iris3d expects: ``xx, yy, zz, yz, xz, xy``.
    """
    x = np.linspace(0.0, length, nx)
    y = np.linspace(-height / 2, height / 2, ny)
    z = np.linspace(-width / 2, width / 2, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()

    second_moment = width * height**3 / 12.0
    moment = load * (length - xx)
    sigma_xx = moment * yy / second_moment
    tau_xy = (load / (2.0 * second_moment)) * ((height / 2) ** 2 - yy**2)

    zero = np.zeros_like(sigma_xx)
    stress = np.stack([sigma_xx, zero, zero, zero, zero, tau_xy], axis=1)
    von_mises = np.sqrt(sigma_xx**2 + 3.0 * tau_xy**2)

    return {
        "positions": np.stack([xx, yy, zz], axis=1).astype(np.float32),
        "stress": stress.astype(np.float32),
        "von_mises": von_mises.astype(np.float32),
    }


def benzene() -> dict[str, np.ndarray]:
    """A benzene ring, C6H6.

    Geometry is derived rather than recalled: the carbons form a regular
    hexagon, whose side length equals its circumradius, so a 1.39 A C-C bond
    puts the carbons at radius 1.39. Hydrogens sit on the same radial lines a
    further 1.09 A out. Everything is planar at ``z = 0``, which is the easiest
    possible thing to check.
    """
    carbon_carbon = 1.39
    carbon_hydrogen = 1.09
    angles = np.arange(6) * (np.pi / 3.0)

    carbons = np.stack(
        [carbon_carbon * np.cos(angles), carbon_carbon * np.sin(angles), np.zeros(6)],
        axis=1,
    )
    hydrogen_radius = carbon_carbon + carbon_hydrogen
    hydrogens = np.stack(
        [hydrogen_radius * np.cos(angles), hydrogen_radius * np.sin(angles), np.zeros(6)],
        axis=1,
    )

    ring = np.stack([np.arange(6), (np.arange(6) + 1) % 6], axis=1)
    c_h = np.stack([np.arange(6), np.arange(6) + 6], axis=1)

    return {
        "positions": np.concatenate([carbons, hydrogens]).astype(np.float32),
        # Atomic numbers: six carbons then six hydrogens.
        "elements": np.array([6] * 6 + [1] * 6, dtype=np.uint8),
        "bonds": np.concatenate([ring, c_h]).astype(np.uint32),
        # biotite BondType: 9 is aromatic (no Kekule assignment), 1 is single.
        "bond_orders": np.array([9] * 6 + [1] * 6, dtype=np.uint8),
    }


def random_cloud(count: int = 250_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Unstructured noise, for measuring throughput rather than correctness."""
    rng = np.random.default_rng(seed)
    return {
        "positions": rng.standard_normal((count, 3)).astype(np.float32),
        "colors": rng.integers(0, 256, size=(count, 3), dtype=np.uint8),
    }


def examples() -> dict[str, dict[str, np.ndarray]]:
    """Every generator except the throughput one, keyed by a display name."""
    return {
        "torus (points)": torus_points(),
        "torus (mesh)": torus_mesh(),
        "cantilever beam": cantilever_beam(),
        "benzene": benzene(),
    }
