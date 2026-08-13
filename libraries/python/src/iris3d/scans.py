"""Measured sample volumes, for looking at rather than for checking.

The opposite trade to :mod:`iris3d.testdata`, and kept apart from it for that
reason. Those generators are formulas, so the right answer is always known and
a sign error or a transposed axis is obvious on sight. These are *recordings*:
nobody knows what the correct image is, so they cannot tell you a renderer is
right — noise and structure look much the same whether or not the transport is
correct.

What they are good for is the question a formula cannot answer: whether a
transfer function, a step count or an absorbance reads well against data with
real texture, real noise and a real dynamic range. A Gaussian blob flatters a
volume renderer; a scan does not.

Both loaders need the ``dev`` dependency group (``scikit-image`` and
``pooch``), and both **download on first use** — scikit-image fetches its own
sample registry into its user cache directory, and every call after that is
local. Nothing is vendored into this repository.
"""

from __future__ import annotations

import numpy as np

from .client import Grid

__all__ = ["cell_nuclei", "ct_head", "mri_head"]

#: The Stanford volume data archive's CTHead, and the hash of the exact archive
#: these loaders were written against.
#:
#: Pinned rather than left open. `pooch` will refuse a file that does not match,
#: which is the point: this is a third-party host, and a volume that quietly
#: changed shape would show up as a rendering bug rather than as a bad download.
_CTHEAD_URL = "https://graphics.stanford.edu/data/voldata/cthead-8bit.tar.gz"
_CTHEAD_SHA256 = "sha256:a0b3414faf1236e94eb3f1f8785f65b51a2c9dfb15dc42e6125f4a2fc5334137"


def _require_skimage():
    """The sample loaders, or an error that says what to install.

    An ``ImportError`` from three frames down names ``pooch`` and not much
    else, and scikit-image's own message points at its install page rather than
    at this project's dependency group.
    """
    try:
        from skimage import data
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "iris3d.scans needs the dev dependency group: "
            "run `uv sync --group dev` in libraries/python"
        ) from exc
    return data


def _fit(dims: tuple[int, int, int], voxel: tuple[float, float, float], size: float):
    """Spacing and origin that put a volume of `dims` at the world origin.

    The *shape* comes from the voxel aspect and the *scale* from `size`, so a
    volume with thick slices stays correctly squashed however large it is drawn.
    Centred rather than corner-placed because the viewport looks at the origin,
    and a scan that has to be hunted for is a poor demonstration.
    """
    spans = [(n - 1) * v for n, v in zip(dims, voxel)]
    longest = max(spans) or 1.0
    scale = size / longest
    spacing = tuple(v * scale for v in voxel)
    origin = tuple(-span * scale / 2.0 for span in spans)
    return spacing, origin


def _to_wire(volume: np.ndarray, clip: float | None) -> np.ndarray:
    """A ``(z, y, x)`` stack as the wire wants it: x fastest once ravelled.

    scikit-image returns slices first. iris3d's grid is declared ``(nx, ny,
    nz)`` and read with z varying fastest, so the axes are reversed rather than
    rolled. Getting this wrong does not fail — it renders a plausible picture of
    a transposed head.

    `clip` is an upper percentile. Measured data has a tail that a plain
    min/max normalisation wastes most of the range on: a handful of bright
    voxels push everything else into the bottom of the ramp and the volume reads
    as uniformly dim. Clipping is a display choice, not a correction, which is
    why it is a parameter.
    """
    values = np.asarray(volume, dtype=np.float32)
    if clip is not None:
        high = float(np.percentile(values, clip))
        low = float(values.min())
        if high > low:
            values = np.clip(values, low, high)
    return np.ascontiguousarray(values.transpose(2, 1, 0)).ravel()


def mri_head(size: float = 8.0, clip: float | None = 99.5) -> tuple[dict[str, np.ndarray], Grid]:
    """An MRI of a head, from scikit-image's sample registry.

    Ten slices of 256x256, which is a real acquisition rather than a truncated
    one: thick-slice MRI trades through-plane resolution for time. The volume is
    therefore a slab, and the anisotropic spacing below is what keeps it from
    rendering as a cube — which also makes this the one sample here that
    exercises a non-uniform grid.

    The slice thickness is *chosen*, not read: the TIFF carries no reliable
    spacing, so the 10:1 aspect is a plausible clinical ratio rather than a
    measurement. Say so before quoting any distance off this render.

    Returns ``(arrays, grid)``. Bind ``intensity`` to a volume's ``density``;
    leaving ``emissive`` unbound makes it glow in proportion to what it blocks,
    which is what a single-field scan wants.
    """
    data = _require_skimage()
    volume = np.asarray(data.brain())
    nz, ny, nx = volume.shape
    dims = (nx, ny, nz)
    # In-plane is fine and through-plane is coarse, which is the whole shape of
    # a thick-slice acquisition.
    spacing, origin = _fit(dims, (1.0, 1.0, 10.0), size)
    return {"intensity": _to_wire(volume, clip)}, Grid(
        dims=dims, origin=origin, spacing=spacing
    )


def ct_head(size: float = 8.0, clip: float | None = None) -> tuple[dict[str, np.ndarray], Grid]:
    """A CT of a head, skull included — the Stanford archive's CTHead.

    Ninety-nine slices of 256x256, which is the volume-rendering benchmark of
    record: nearly every paper on the subject since the late eighties has shown
    this head. Unlike :func:`mri_head` it has the depth to be worth marching
    through, and unlike :func:`cell_nuclei` it has a hard, bright, thin
    structure inside a soft one — which is the case a smooth analytic blob
    cannot pose at all.

    Downloaded from `graphics.stanford.edu` rather than from scikit-image's
    registry, and the archive hash is pinned; see :data:`_CTHEAD_SHA256`.

    `clip` defaults to *off*, unlike the other two. The bright end of a CT is
    the bone, and it is a small fraction of the voxels — so an upper-percentile
    clip would crush exactly the structure this dataset is worth having for.

    The slice spacing is **chosen**, not read: these are bare 8-bit TIFFs with
    no metadata, and the rescaling to 8 bits has already discarded the
    Hounsfield mapping. 1:1:2 is a plausible clinical ratio. Nothing metric
    should be read off this render.

    Returns ``(arrays, grid)`` with one field, ``intensity``. There is no
    separate bone field to bind to ``emissive``, because any threshold that
    produced one would be invented rather than measured — pin the colour range
    in the Actors tab instead, which is what that control is for.
    """
    try:
        import pooch
        from skimage.io import imread
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "iris3d.scans needs the dev dependency group: "
            "run `uv sync --group dev` in libraries/python"
        ) from exc

    paths = pooch.retrieve(
        url=_CTHEAD_URL,
        known_hash=_CTHEAD_SHA256,
        processor=pooch.Untar(),
    )
    # Zero-padded, so lexicographic order is slice order. Worth being explicit
    # about: an unpadded scheme would put slice 10 before slice 2 and shuffle
    # the head into nonsense that still renders.
    volume = np.stack([np.asarray(imread(path)) for path in sorted(paths)])
    nz, ny, nx = volume.shape
    dims = (nx, ny, nz)
    spacing, origin = _fit(dims, (1.0, 1.0, 2.0), size)
    return {"intensity": _to_wire(volume, clip)}, Grid(
        dims=dims, origin=origin, spacing=spacing
    )


def cell_nuclei(size: float = 8.0, clip: float | None = 99.5) -> tuple[dict[str, np.ndarray], Grid]:
    """Fluorescence microscopy of cells: membranes and nuclei, separately.

    Sixty slices rather than ten, so unlike :func:`mri_head` this is a volume
    with genuine depth to march through. Its real value is that the two channels
    are two *different measured quantities* of the same specimen — so binding
    them to ``density`` and ``emissive`` is not a contrivance to show the
    feature off, it is the thing the feature is for: the membranes occlude, the
    nuclei glow inside them.

    Voxel size is the dataset's own, 0.26 x 0.26 x 0.29 micrometre, so the
    proportions here are measured rather than chosen.

    Returns ``(arrays, grid)`` with ``membrane`` and ``nuclei``.
    """
    data = _require_skimage()
    volume = np.asarray(data.cells3d())
    nz, _channels, ny, nx = volume.shape
    dims = (nx, ny, nz)
    spacing, origin = _fit(dims, (0.26, 0.26, 0.29), size)
    return {
        "membrane": _to_wire(volume[:, 0], clip),
        "nuclei": _to_wire(volume[:, 1], clip),
    }, Grid(dims=dims, origin=origin, spacing=spacing)
