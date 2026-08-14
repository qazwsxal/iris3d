"""Loading molecular and protein structures via biotite.

This is the escape hatch working as intended: iris3d has no PDB or mmCIF
parser and does not need one. A client parses locally with whatever library
suits it and uploads plain arrays.

Residues and chains are plain arrays too. A dense ``residue_index`` per atom
says which residue it belongs to, and arrays keyed on that index carry what
each residue is called and how it is numbered. Text is dictionary-encoded the
same way, so no array of strings ever grows with the atom count. Names follow
biotite's annotation categories, for the same reason bond orders follow its ``BondType``
enum: those names are close to a faithful reading of the mmCIF data model, so
"follow biotite" is in practice "follow mmCIF" and a wrapper in another
language reaches the same fields.

Secondary structure is the one deliberate exception — see :data:`SSE_CODES`.

biotite is a *development* dependency, so it is imported lazily — installing
iris3d does not drag in biotite (or its LGPL-licensed `biotraj` transitive
dependency). Call anything here without it and you get a clear error rather
than an ImportError from three frames down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biotite.structure import AtomArray

__all__ = [
    "BOND_TYPES",
    "SSE_CODES",
    "arrays_from_atoms",
    "atomic_numbers",
    "fetch",
    "load_structure",
    "residue",
]

# Symbols in atomic-number order; index + 1 is Z.
_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
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

#: Secondary-structure codes, one per residue. DSSP's eight states, which is
#: what the wire carries — see ``scene.proto``.
#:
#: Eight rather than helix/strand/coil, because cartoon rendering that draws a
#: 3-10 helix differently from an alpha helix cannot get there from three
#: letters, and the width costs nothing: it is one integer either way.
SSE_CODES = {
    0: "none",
    1: "alpha helix",
    2: "beta bridge",
    3: "strand",
    4: "3-10 helix",
    5: "pi helix",
    6: "turn",
    7: "bend",
}

#: biotite's ``annotate_sse`` implements P-SEA, which returns three states.
#: They map up into the wider space and lose nothing they had; the states P-SEA
#: cannot tell apart simply never appear.
#:
#: This is the one place iris3d deliberately does *not* adopt biotite's
#: convention. Swapping in a real DSSP assignment later — biotite's own
#: ``application.dssp``, or codes read straight out of an mmCIF file — changes
#: what fills this array and nothing about the format.
_PSEA_TO_DSSP = {"a": 1, "b": 3, "c": 0}


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


def atomic_numbers(symbols: np.ndarray | list[str]) -> np.ndarray:
    """Maps element symbols to atomic numbers, as uint8.

    Unrecognised or blank symbols become 0, which the renderer draws in the
    fallback colour rather than rejecting.
    """
    return np.array(
        [_BY_SYMBOL.get(str(symbol).strip().upper(), 0) for symbol in symbols],
        dtype=np.uint8,
    )


def _encoded(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """Dictionary-encodes a text column into an index array and its distinct values.

    Returns ``{name + "_index": ..., name: ...}`` — the index is one integer per
    element, and ``name`` holds only the values that actually occur.

    **This is what keeps text off the critical path.** A string array travels
    whole in the upload header, which is a single gRPC message, so a per-atom
    one grows without bound: ``atom_name`` measures 3.8 bytes an atom, which
    reaches the server's 8 MiB limit at about 2.2 million atoms and fails at the
    transport rather than anywhere helpful.

    Text columns are all low-cardinality, which is what makes this work rather
    than merely shrink things. 1HHO has 85 distinct atom names across 2396
    atoms, and that count saturates — a system ten times the size has much the
    same handful of names. So the index array is ordinary numeric data that
    chunks like any other, and the header carries the distinct values alone.

    It is also the same shape as everything around it: a dense index plus an
    array keyed on it, exactly as ``residue_index`` works one level up.
    """
    distinct, index = np.unique(np.asarray(values), return_inverse=True)
    # The narrowest integer that indexes the dictionary. uint16 covers 65535
    # distinct values, which no real name column approaches.
    dtype = np.uint16 if len(distinct) <= np.iinfo(np.uint16).max else np.uint32
    return {
        f"{name}_index": index.astype(dtype),
        name: distinct.astype(object),
    }


def _dense_index(starts: np.ndarray, count: int) -> np.ndarray:
    """Turns group start offsets into a dense group index per element.

    ``starts`` is where each group begins, so the gaps between successive
    entries are the group sizes and repeating each group's ordinal by its size
    gives one index per element.
    """
    sizes = np.diff(np.append(starts, count))
    return np.repeat(np.arange(len(starts), dtype=np.uint32), sizes)


def _secondary_structure(atoms: AtomArray, residues: int) -> np.ndarray | None:
    """One DSSP code per residue, or ``None`` if it cannot be assigned.

    biotite's P-SEA assignment covers amino acids only, so a structure with
    waters, ligands or nucleic acids gets back fewer codes than it has
    residues. Those are scattered back onto the residues they belong to and
    everything else stays 0 — unassigned, which is exactly what it is.

    Returns ``None`` rather than guessing when the counts cannot be reconciled.
    A misaligned array is worse than an absent one: it would paint every
    residue with its neighbour's structure and look plausible doing it.
    """
    struc = _biotite()

    try:
        codes = struc.annotate_sse(atoms)
    except Exception:
        # P-SEA needs backbone geometry. A ligand, a bare CCD template or a
        # structure with no CA atoms has none, which is not an error worth
        # raising — it simply has no secondary structure.
        return None

    sse = np.zeros(residues, dtype=np.uint8)
    mapped = np.array(
        [_PSEA_TO_DSSP.get(str(code), 0) for code in codes], dtype=np.uint8
    )

    if len(mapped) == residues:
        return mapped

    # Fewer codes than residues: they belong to the amino-acid residues, in
    # order. Find which those are by looking at each residue's first atom.
    starts = struc.get_residue_starts(atoms)
    amino = np.flatnonzero(struc.filter_amino_acids(atoms)[starts])
    if len(mapped) != len(amino):
        return None

    sse[amino] = mapped
    return sse


def _hierarchy(atoms: AtomArray) -> dict[str, np.ndarray]:
    """Residue and chain structure, as per-atom indices plus side arrays.

    Two pieces, per the wire contract. A dense index per atom says which group
    it belongs to, and arrays keyed on that index carry each group's own
    properties.

    The dense index is the part that matters. Author residue numbering is not a
    key: ``res_id`` repeats across chains, skips, goes negative and carries
    insertion codes, so 100, 100A and 100B are three residues wearing two
    numbers. It travels as a label inside the residue side arrays, and
    ``residue_index`` is what anything actually keys on.
    """
    struc = _biotite()
    count = atoms.array_length()

    residue_starts = struc.get_residue_starts(atoms)

    # Chains are keyed on the identifier, not on contiguous runs of it.
    #
    # biotite's `get_chain_starts` breaks a chain wherever `chain_id` changes,
    # which for a real file breaks it more often than anyone means. 1HHO holds
    # protein A, protein B, a haem in each, then waters in each — six runs over
    # two chains, reported as `['A', 'B', 'A', 'B', 'A', 'B']`. Colour-by-chain
    # over that gives six colours for a two-chain structure.
    #
    # So `chain_index` indexes distinct identifiers. It is no longer monotonic
    # along the atoms, which is the honest consequence: a chain is a thing the
    # file names, not a stretch of the file.
    chain_ids, chain_index = np.unique(
        np.asarray(atoms.chain_id), return_inverse=True
    )
    chain_index = chain_index.astype(np.uint32)

    arrays: dict[str, np.ndarray] = {
        "residue_index": _dense_index(residue_starts, count),
        "chain_index": chain_index,
        # Per residue, keyed on residue_index.
        "residue_res_id": np.asarray(atoms.res_id)[residue_starts].astype(np.int32),
        "residue_hetero": np.asarray(atoms.hetero)[residue_starts].astype(np.uint8),
        # Which chain each residue sits in, so the two levels join without
        # anyone re-deriving it from the per-atom arrays.
        "residue_chain_index": chain_index[residue_starts],
        # Per chain, keyed on chain_index. Not dictionary-encoded: the chain
        # index already is the dictionary, and there are a handful of chains.
        "chain_id": chain_ids.astype(object),
    }

    # Text columns travel dictionary-encoded, so nothing on the wire grows with
    # the number of atoms. See `_encoded`.
    arrays.update(_encoded("atom_name", atoms.atom_name))
    arrays.update(_encoded("residue_name", np.asarray(atoms.res_name)[residue_starts]))
    arrays.update(
        _encoded("residue_ins_code", np.asarray(atoms.ins_code)[residue_starts])
    )

    sse = _secondary_structure(atoms, len(residue_starts))
    if sse is not None:
        arrays["residue_sse"] = sse

    return arrays


def arrays_from_atoms(
    atoms: AtomArray,
    *,
    connect: bool = True,
    hierarchy: bool = True,
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

    Residue and chain hierarchy comes too, unless ``hierarchy`` is off. Per
    atom: ``residue_index``, ``chain_index`` and ``atom_name_index``. Per
    residue, keyed on ``residue_index``: ``residue_res_id``,
    ``residue_hetero``, ``residue_chain_index``, ``residue_sse``,
    ``residue_name_index`` and ``residue_ins_code_index``. Per chain, keyed on
    ``chain_index``: ``chain_id``.

    Text is dictionary-encoded, so each ``*_index`` above reads into a matching
    array of distinct values — ``atom_name``, ``residue_name``,
    ``residue_ins_code`` — holding only the names that occur. Nothing textual
    grows with the atom count; see :func:`_encoded` for why that matters.

    Turn ``hierarchy`` off for a bare molecule where the grouping says nothing,
    such as a single CCD component.

    Note that ``residue_res_id`` is a *label*, not a key. Author numbering
    repeats across chains and skips; ``residue_index`` is what to key on.
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

    if hierarchy:
        arrays.update(_hierarchy(atoms))

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


def load_structure(
    path: str, *, model: int = 1, connect: bool = True, hierarchy: bool = True
) -> dict[str, np.ndarray]:
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
    return arrays_from_atoms(atoms, connect=connect, hierarchy=hierarchy)


def fetch(
    pdb_id: str,
    *,
    directory: str | None = None,
    file_format: Literal["pdb", "cif"] = "pdb",
    model: int = 1,
    connect: bool = True,
    hierarchy: bool = True,
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
    return load_structure(
        str(path), model=model, connect=connect, hierarchy=hierarchy
    )


def residue(name: str) -> dict[str, np.ndarray]:
    """Arrays for one component of the Chemical Component Dictionary.

    Offline: biotite ships the CCD, so ``residue("HEM")`` gives real heme
    geometry with real bonds and no network access. Useful for exercising the
    molecular path without a structure file to hand.
    """
    _biotite()
    from biotite.structure import info

    template = info.residue(name)
    if template is None:
        raise KeyError(f"no component named {name!r} in the CCD")
    return arrays_from_atoms(template)
