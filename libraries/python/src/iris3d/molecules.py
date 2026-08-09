"""Loading molecular and protein structures via biotite.

This is the escape hatch working as intended: iris3d has no PDB or mmCIF
parser and does not need one. A client parses locally with whatever library
suits it and uploads plain arrays.

biotite is a *development* dependency, so it is imported lazily — installing
iris3d does not drag in biotite (or its LGPL-licensed `biotraj` transitive
dependency). Call anything here without it and you get a clear error rather
than an ImportError from three frames down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray

__all__ = [
    "BOND_TYPES",
    "arrays_from_atoms",
    "atomic_numbers",
    "fetch",
    "load_structure",
    "residue",
]

# Symbols in atomic-number order; index + 1 is Z.
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_BY_SYMBOL = {symbol.upper(): index + 1 for index, symbol in enumerate(_SYMBOLS)}

#: Bond types, matching biotite's `BondType` enum, which iris3d adopts as its
#: wire convention. Passed through unchanged rather than folded into a
#: single/double/triple/aromatic scheme, which would lose the distinction
#: between an aromatic single and an aromatic double, and have nowhere to put
#: coordination bonds at all.
BOND_TYPES = {
    0: "any",
    1: "single",
    2: "double",
    3: "triple",
    4: "quadruple",
    5: "aromatic single",
    6: "aromatic double",
    7: "aromatic triple",
    8: "coordination",
    9: "aromatic",
}


def _biotite() -> Any:
    """Imports biotite, or explains why it is missing."""
    try:
        import biotite.structure as struc
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError(
            "biotite is required to read molecular structures and is a "
            "development dependency of iris3d. Install it with "
            "`uv add --dev biotite`, or `pip install biotite`."
        ) from err
    return struc


def atomic_numbers(symbols: "np.ndarray | list[str]") -> np.ndarray:
    """Maps element symbols to atomic numbers, as uint8.

    Unrecognised or blank symbols become 0, which the renderer draws in the
    fallback colour rather than rejecting.
    """
    return np.array(
        [_BY_SYMBOL.get(str(symbol).strip().upper(), 0) for symbol in symbols],
        dtype=np.uint8,
    )


def arrays_from_atoms(
    atoms: "AtomArray",
    *,
    connect: bool = True,
) -> dict[str, np.ndarray]:
    """Converts a biotite ``AtomArray`` into arrays ready to upload.

    Produces ``positions``, ``elements``, and — when connectivity is available
    — ``bonds`` and ``bond_orders``, which is what iris3d recognises as a
    molecule. Any per-atom numeric annotation biotite carries (``b_factor``,
    ``occupancy``, ``charge``) is included as a scalar field, so it can be
    colour-mapped.

    Structures read from PDB and mmCIF usually arrive with no bonds at all.
    With ``connect`` set, connectivity is inferred from residue templates,
    which is what makes ball-and-stick possible.

    Residue and chain hierarchy is **dropped**: the wire format has nowhere to
    put it yet, so a cartoon actor is not reachable from here. What uploads is
    a flat set of atoms and bonds.
    """
    struc = _biotite()

    coords = np.asarray(atoms.coord, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"expected an AtomArray with [n, 3] coordinates, got {coords.shape}. "
            "For a trajectory (AtomArrayStack), pick one model first."
        )

    arrays: dict[str, np.ndarray] = {
        "positions": coords,
        "elements": atomic_numbers(atoms.element),
    }

    bonds = atoms.bonds
    if bonds is None and connect:
        bonds = struc.connect_via_residue_names(atoms)
    if bonds is not None:
        table = bonds.as_array()
        if len(table):
            arrays["bonds"] = table[:, :2].astype(np.uint32)
            # biotite's BondType values go straight onto the wire; see
            # BOND_TYPES.
            arrays["bond_orders"] = table[:, 2].astype(np.uint8)

    # Per-atom numeric annotations become colour-mappable fields.
    for name in ("b_factor", "occupancy", "charge"):
        if name not in atoms.get_annotation_categories():
            continue
        values = np.asarray(atoms.get_annotation(name))
        if not np.issubdtype(values.dtype, np.number):
            continue
        if np.allclose(values, values.flat[0]):
            # A constant column carries no information and would autoscale to
            # a single flat colour.
            continue
        arrays[name] = values.astype(np.float32)

    return arrays


#: Per-atom annotations worth asking for. biotite's readers skip these unless
#: named, and B-factor in particular is the field most worth colouring a
#: protein by, so leaving it behind would waste the trip.
_EXTRA_FIELDS = ["b_factor", "occupancy", "charge"]


def load_structure(path: str, *, model: int = 1, connect: bool = True) -> dict[str, np.ndarray]:
    """Reads a structure file (PDB, mmCIF, BinaryCIF, MOL/SDF...) for upload.

    ``model`` selects one frame from a multi-model file such as an NMR
    ensemble; biotite numbers models from 1.
    """
    _biotite()
    import biotite.structure.io as strucio

    try:
        atoms = strucio.load_structure(path, model=model, extra_fields=_EXTRA_FIELDS)
    except TypeError:
        # Not every format reader accepts extra_fields; the structure itself
        # still loads, just without the optional annotations.
        atoms = strucio.load_structure(path, model=model)
    return arrays_from_atoms(atoms, connect=connect)


def fetch(
    pdb_id: str,
    *,
    directory: str | None = None,
    file_format: str = "pdb",
    model: int = 1,
    connect: bool = True,
) -> dict[str, np.ndarray]:
    """Downloads a structure from RCSB and returns arrays ready to upload.

    **This makes a network request.** Everything else in this module works
    offline; this is the one function that does not, which is why it is
    separate from :func:`load_structure` rather than a path-or-id argument to
    it.

    Files are cached in ``directory`` (a temporary directory by default), so a
    repeated call for the same entry does not re-download.
    """
    _biotite()
    import tempfile

    from biotite.database import rcsb

    directory = directory or tempfile.gettempdir()
    path = rcsb.fetch(pdb_id, file_format, directory)
    return load_structure(str(path), model=model, connect=connect)


def residue(name: str) -> dict[str, np.ndarray]:
    """Arrays for one component of the Chemical Component Dictionary.

    Offline: biotite ships the CCD, so ``residue("HEM")`` gives real heme
    geometry with real bonds and no network access. Useful for exercising the
    molecular path without a structure file to hand.
    """
    _biotite()
    import biotite.structure.info as info

    template = info.residue(name)
    if template is None:
        raise KeyError(f"no component named {name!r} in the CCD")
    return arrays_from_atoms(template)
