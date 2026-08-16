"""Electron-density and cryo-EM maps, as grids ready to upload.

A map is the *evidence* a structure was built from, and the reason to load one
alongside a model is to see the two together: the ribbon inside the density,
with the density in front of it correctly blocking the light coming off it.

**Coordinates are left alone.** A map and the model fitted into it already share
a frame, in ångströms, so neither is re-centred or rescaled here — that is the
whole point, and it is the one way this module differs from :mod:`iris3d.scans`,
whose samples are anonymous volumes placed wherever is convenient.

Only MRC/CCP4 is read, which is what EMDB serves and what every refinement
program writes. The reader is about sixty lines because the format is a fixed
1024-byte header and a dense block of samples; a dependency would cost more than
it saved.
"""

from __future__ import annotations

import gzip
import struct
from typing import BinaryIO

import numpy as np

from .client import Grid

__all__ = ["MapHeader", "downsample", "fetch_emdb", "load_map", "read_map"]

#: MRC mode to numpy dtype. Modes 3 and 4 are complex transforms, which are not
#: densities and are refused rather than reinterpreted.
_MODES = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    6: np.uint16,
    12: np.float16,
}

#: Bytes of MRC header before the data (or before the extended header).
_HEADER = 1024


class MapHeader:
    """What an MRC header says about where the samples sit.

    Only the fields that place the grid in space. Everything else in the header
    — symmetry, statistics, the label block — describes the experiment rather
    than the geometry, and nothing here needs it.
    """

    def __init__(self, raw: bytes) -> None:
        # Counts along columns, rows and sections: the *storage* order, which is
        # not necessarily x, y, z. `axes` below says which is which.
        self.counts = struct.unpack_from("<3i", raw, 0)
        self.mode = struct.unpack_from("<i", raw, 12)[0]
        self.starts = struct.unpack_from("<3i", raw, 16)
        # The sampling grid the cell is divided into. Voxel size comes from this
        # and the cell, *not* from the counts: a map can be a sub-box of its own
        # unit cell, and dividing by the counts then gives a voxel that is wrong
        # by the ratio of the two.
        self.sampling = struct.unpack_from("<3i", raw, 28)
        self.cell = struct.unpack_from("<3f", raw, 40)
        # Which crystal axis each storage axis is: 1 = x, 2 = y, 3 = z.
        self.axes = struct.unpack_from("<3i", raw, 64)
        self.symmetry_bytes = struct.unpack_from("<i", raw, 92)[0]
        self.origin = struct.unpack_from("<3f", raw, 196)
        self.stamp = raw[208:212]

    def voxel(self) -> tuple[float, float, float]:
        """Sample spacing along x, y and z, in ångströms."""
        return tuple(
            cell / count if count else 1.0
            for cell, count in zip(self.cell, self.sampling)
        )

    def world_origin(self) -> tuple[float, float, float]:
        """Where sample (0, 0, 0) sits, in ångströms.

        Two conventions exist and both are in the wild. The `ORIGIN` words are
        authoritative when they are set; otherwise the start indices give the
        offset in samples and it is scaled by the voxel. Preferring `ORIGIN`
        matches what EMDB writes and what every viewer reads.
        """
        if any(abs(value) > 1e-6 for value in self.origin):
            return self.origin
        voxel = self.voxel()
        # The starts are in storage order, so they have to be put back into
        # x, y, z before they can be scaled.
        starts = [0.0, 0.0, 0.0]
        for storage, axis in enumerate(self.axes):
            starts[axis - 1] = self.starts[storage]
        return tuple(start * size for start, size in zip(starts, voxel))


def read_map(stream: BinaryIO) -> tuple[np.ndarray, Grid]:
    """Reads an MRC/CCP4 map, returning a ``(z, y, x)`` array and its grid.

    The array is oriented so that its axes are z, y, x whatever order the file
    stored them in — MRC records that in `MAPC`/`MAPR`/`MAPS`, and a map written
    with the sections along x renders as a transposed blob if it is ignored.
    """
    raw = stream.read(_HEADER)
    if len(raw) < _HEADER:
        raise ValueError("truncated MRC header")
    header = MapHeader(raw)

    if header.stamp not in (b"MAP ", b"MAP\x00"):
        raise ValueError(
            f"not an MRC/CCP4 map: expected a 'MAP ' stamp, found {header.stamp!r}"
        )
    dtype = _MODES.get(header.mode)
    if dtype is None:
        raise ValueError(
            f"MRC mode {header.mode} is not a density map "
            f"(supported: {sorted(_MODES)})"
        )

    # The extended header carries symmetry operators and per-slice metadata.
    # Neither places the grid, so it is skipped rather than parsed.
    if header.symmetry_bytes:
        stream.read(header.symmetry_bytes)

    columns, rows, sections = header.counts
    values = np.frombuffer(
        stream.read(columns * rows * sections * np.dtype(dtype).itemsize), dtype=dtype
    )
    if values.size != columns * rows * sections:
        raise ValueError(
            f"map holds {values.size} samples, header declares "
            f"{columns * rows * sections}"
        )
    # Sections are outermost and columns innermost, always.
    values = values.reshape(sections, rows, columns)
    # Only when it is not already float32, which mode 2 is. A published map runs
    # to hundreds of millions of samples — EMD-53922 is 640 cubed, a gigabyte —
    # and an unconditional `astype` would copy the whole of it to change
    # nothing.
    if values.dtype != np.float32:
        values = values.astype(np.float32)

    # Put the axes into z, y, x. `axes` maps storage axis to crystal axis, and
    # the array's axes run (sections, rows, columns) = storage 2, 1, 0.
    storage_of_axis = {axis: storage for storage, axis in enumerate(header.axes)}
    if sorted(storage_of_axis) != [1, 2, 3]:
        raise ValueError(f"MRC axis order {header.axes} is not a permutation of x, y, z")
    values = values.transpose(
        *(2 - storage_of_axis[axis] for axis in (3, 2, 1))
    )

    shape_zyx = values.shape
    grid = Grid(
        dims=(shape_zyx[2], shape_zyx[1], shape_zyx[0]),
        origin=header.world_origin(),
        spacing=header.voxel(),
    )
    # As above: the usual axis order makes the transpose a no-op, so this is
    # already contiguous and asking again would copy a gigabyte for nothing.
    if not values.flags["C_CONTIGUOUS"]:
        values = np.ascontiguousarray(values)
    return values, grid


def downsample(values: np.ndarray, grid: Grid, factor: int) -> tuple[np.ndarray, Grid]:
    """Averages `factor`³ blocks together, keeping the grid in the same place.

    A published map is routinely 300³ or more, which is 27 million samples and
    a 3D texture to match. Halving each axis is eight times less of both and
    costs a resolution nobody is measuring off a picture.

    Block **mean** rather than stride. Striding throws away seven eighths of the
    signal and aliases the rest; averaging is what the sampling theorem asks for
    and is one line either way.

    The origin shifts by half a block, because the mean of a block sits at its
    centre rather than at its first corner. Getting that wrong slides the map off
    the model by up to a voxel, which looks like a fitting error.
    """
    if factor <= 1:
        return values, grid
    depth, height, width = values.shape
    trimmed = values[
        : depth // factor * factor,
        : height // factor * factor,
        : width // factor * factor,
    ]
    if 0 in trimmed.shape:
        raise ValueError(f"downsampling by {factor} would leave nothing of {values.shape}")
    blocks = trimmed.reshape(
        trimmed.shape[0] // factor,
        factor,
        trimmed.shape[1] // factor,
        factor,
        trimmed.shape[2] // factor,
        factor,
    )
    reduced = blocks.mean(axis=(1, 3, 5), dtype=np.float32)

    spacing = tuple(size * factor for size in grid.spacing)
    origin = tuple(
        start + size * (factor - 1) / 2.0 for start, size in zip(grid.origin, grid.spacing)
    )
    return np.ascontiguousarray(reduced), Grid(
        dims=(reduced.shape[2], reduced.shape[1], reduced.shape[0]),
        origin=origin,
        spacing=spacing,
    )


def crop(
    values: np.ndarray,
    grid: Grid,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> tuple[np.ndarray, Grid]:
    """Cuts the map down to a world-space box, in ångströms.

    A published map is padded with solvent — the box is routinely twice the
    span of what is in it, and EMD-3061 is 253 Å across for a 135 Å complex.
    Uploading the padding costs texture and, worse, drags the camera's framing
    out to the empty box so the molecule renders small in the middle of nothing.

    Bounds are clamped to the map, so asking for more than there is gives what
    there is rather than an error.
    """
    depth, height, width = values.shape
    limits = (width, height, depth)
    starts, stops = [], []
    for axis in range(3):
        first = int(np.floor((low[axis] - grid.origin[axis]) / grid.spacing[axis]))
        last = int(np.ceil((high[axis] - grid.origin[axis]) / grid.spacing[axis])) + 1
        first = max(0, min(first, limits[axis] - 1))
        last = max(first + 1, min(last, limits[axis]))
        starts.append(first)
        stops.append(last)

    # The array is (z, y, x) and the bounds are (x, y, z).
    cut = values[starts[2] : stops[2], starts[1] : stops[1], starts[0] : stops[0]]
    origin = tuple(
        grid.origin[axis] + starts[axis] * grid.spacing[axis] for axis in range(3)
    )
    return np.ascontiguousarray(cut), Grid(
        dims=(cut.shape[2], cut.shape[1], cut.shape[0]),
        origin=origin,
        spacing=grid.spacing,
    )


def _to_wire(values: np.ndarray, floor: float | None) -> np.ndarray:
    """A ``(z, y, x)`` stack as ``(nx, ny, nz)``, optionally with noise cut.

    The axis reversal is :mod:`iris3d.scans`'s and is load-bearing in the same
    way: getting it wrong does not fail, it renders a convincing picture of a
    transposed map.

    `floor` is a contour level in the map's own units — the number a viewer
    calls "level" or "sigma". Everything below it becomes zero, which is what
    turns a box of noise into a molecule. Without it the solvent is a uniform
    haze that swamps whatever is inside it.
    """
    values = np.asarray(values, dtype=np.float32)
    if floor is not None:
        values = np.where(values < floor, 0.0, values - floor)
    # Normalised to 0..1, so a volume actor's `opacity` means the same thing
    # from one map to the next. Map units are arbitrary — they depend on the
    # scaling of the reconstruction — so an absorbance quoted against them would
    # have to be retuned for every entry.
    peak = float(values.max())
    if peak > 0.0:
        values = values / peak
    # Kept three-dimensional. A grid's shape is something the array already
    # knows, so `volume` and `contour` both declare `[0, 0, 0]` and read it off
    # the binding; ravelling it threw that away and left `dims` to state it
    # separately, which is the parameter that could disagree with its own data.
    return np.ascontiguousarray(values.transpose(2, 1, 0))


def load_map(
    path: str,
    *,
    factor: int = 1,
    floor: float | None = None,
    sigma: float | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> tuple[dict[str, np.ndarray], Grid]:
    """Reads a map file for upload, returning ``(arrays, grid)``.

    Bind ``density`` to a volume actor's ``density``, and the grid's ``origin``
    and ``spacing`` to the matching inputs. There is no ``dims`` to bind: the
    array is ``(nx, ny, nz)`` and the grid's shape is read off it. Coordinates
    stay in ångströms, so a structure uploaded from the same entry lands inside
    it.

    `factor` averages blocks together; see :func:`downsample`. `sigma` sets the
    contour level as a multiple of the map's own standard deviation, which is
    how contour levels are quoted — 2σ or 3σ is the usual range for a cryo-EM
    map. `floor` sets it in raw units instead, for a map whose level you already
    know. Give one or the other.

    Gzip is detected by its magic number rather than by the file extension,
    because EMDB serves ``.map.gz`` and people rename things.
    """
    if floor is not None and sigma is not None:
        raise ValueError("give either `floor` or `sigma`, not both")

    with open(path, "rb") as handle:
        opener = gzip.open if handle.read(2) == b"\x1f\x8b" else None
    with (gzip.open(path, "rb") if opener else open(path, "rb")) as stream:
        values, grid = read_map(stream)

    values, grid = downsample(values, grid, factor)
    # The contour level is worked out before cropping, so it is a property of
    # the map rather than of whatever box happened to be asked for.
    if sigma is not None:
        floor = float(values.mean() + sigma * values.std())
    if bounds is not None:
        values, grid = crop(values, grid, *bounds)
    return {"density": _to_wire(values, floor)}, grid


#: Where EMDB serves its maps. The id is zero-padded to four digits in the
#: directory name and bare in the file name, which is the one awkward part of
#: the layout.
_EMDB_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{id}/map/emd_{id}.map.gz"

#: Where EMDB answers questions about an entry, including the contour level.
_EMDB_API = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{id}"


def recommended_contour(emd_id: str | int) -> float | None:
    """The contour level the depositors recommend, or ``None`` if none is given.

    **This is better than any threshold worked out here.** A sigma multiple is a
    guess from the histogram; the deposited level is the one the people who
    built the model actually looked at it at, and it is what every viewer opens
    the entry with. Guessing is the fallback, not the default.

    Returns ``None`` rather than raising when the entry has no level or the
    lookup fails, so an offline caller falls back to sigma instead of stopping.
    """
    import json
    import urllib.request

    number = str(emd_id).upper().removeprefix("EMD-")
    try:
        with urllib.request.urlopen(_EMDB_API.format(id=number), timeout=30) as reply:
            entry = json.load(reply)
        contours = entry["map"]["contour_list"]["contour"]
    except Exception:
        return None

    # Several may be listed; the primary one is flagged, and the first is the
    # convention when none is.
    primary = next((c for c in contours if c.get("primary")), None) or contours[0]
    level = primary.get("level")
    return None if level is None else float(level)


def fetch_emdb(
    emd_id: str | int,
    *,
    directory: str | None = None,
    factor: int = 1,
    floor: float | None = None,
    sigma: float | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> tuple[dict[str, np.ndarray], Grid]:
    """Downloads a map from EMDB and returns it ready to upload.

    **This makes a network request**, and a map is tens of megabytes — far more
    than a structure file. Downloads are cached in `directory` (a temporary
    directory by default), so a repeated call for the same entry does not
    re-fetch.

    With neither `floor` nor `sigma`, the **deposited contour level** is used —
    see :func:`recommended_contour`, and prefer leaving both unset. A sigma
    multiple is a guess from the histogram; the deposited level is what the
    people who built the model looked at it at.

    `emd_id` may be given as ``"EMD-20026"``, ``"20026"`` or ``20026``.
    """
    import os
    import tempfile
    import urllib.request

    number = str(emd_id).upper().removeprefix("EMD-")
    if not number.isdigit():
        raise ValueError(f"not an EMDB id: {emd_id!r}")

    if floor is None and sigma is None:
        floor = recommended_contour(number)
        if floor is None:
            # No level deposited, or no network. Fall back to a guess and say
            # which one, so a picture is never silently contoured at a threshold
            # nobody chose.
            sigma = 3.0

    directory = directory or tempfile.gettempdir()
    path = os.path.join(directory, f"emd_{number}.map.gz")
    if not os.path.exists(path):
        url = _EMDB_URL.format(id=number)
        # To a temporary name first, so an interrupted download does not leave
        # a truncated file that every later call then trusts.
        partial = path + ".part"
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, path)

    return load_map(path, factor=factor, floor=floor, sigma=sigma, bounds=bounds)
