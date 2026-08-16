"""A protein cartoon inside the cryo-EM map it was built from.

This is the test the moment backend exists for. Two objects share one
coordinate frame — nothing is re-centred, because a map and the model fitted
into it already agree in ångströms:

- the **structure**, as an *opaque* cartoon. It goes through Bevy's ordinary
  opaque pass, so it is lit and it writes depth;
- the **map**, as a volume that deposits absorbance into the moment buffer.

The composition is the point. The accumulation truncates every interval at the
opaque depth, so density *in front of* the ribbon blocks the light coming off
it and density *behind* it contributes nothing — correct from every angle, at
every pixel, with nothing sorted. Alpha blending cannot do this: the map is a
participating medium sampled along the ray rather than a surface, so there is no
single depth to sort it against, and the ribbon threads through it — in front of
some samples and behind others — within a single pixel.

Run against a running app::

    cargo run
    uv run python density_demo.py

Defaults to gamma-secretase: EMD-3061 at 3.4 Å with PDB 5A63 fitted into it.
Pass a different pair as ``<pdb-id> <emd-id>``.
"""

from __future__ import annotations

import sys

import numpy as np

import iris3d
from iris3d import density, molecules

#: The default pair. Chosen for being honest rather than pretty: a real
#: membrane-protein complex at a resolution where the map is visibly a map, and
#: the smallest EMDB download of the candidates at 22 MB.
DEFAULT_PDB = "5a63"
DEFAULT_EMD = "3061"

#: The six arrays the cartoon filter binds, as input id -> the name `molecules`
#: gives it.
ROLES = {
    "positions": "positions",
    "residue_index": "residue_index",
    "atom_name_index": "atom_name_index",
    "atom_name": "atom_name",
    "residue_sse": "residue_sse",
    "residue_chain_index": "residue_chain_index",
}

#: How far past the model to keep the map, in ångströms. Enough to show the
#: density standing off the ribbon rather than clipped tight against it.
MARGIN = 12.0


def main():
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDB
    emd_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EMD

    print(f"fetching {pdb_id} from RCSB...")
    structure = molecules.fetch(pdb_id, file_format="cif")
    positions = structure["positions"]
    low = tuple(positions.min(axis=0) - MARGIN)
    high = tuple(positions.max(axis=0) + MARGIN)

    print(f"fetching EMD-{emd_id} from EMDB (tens of MB)...")
    # Full resolution: `factor=1` keeps every sample the deposition has. The
    # crop is what makes that affordable — the map's own box is nearly twice the
    # span of what is in it, so cutting to the model plus a margin takes 673k
    # samples where the whole box would be 5.8M, and the padding would drag the
    # camera's framing out to empty space besides.
    #
    # 3 sigma is the contour: at 2 the solvent is still above it and the molecule
    # sits in a haze of speckle.
    volume, grid = density.fetch_emdb(
        emd_id, factor=1, sigma=3.0, bounds=(low, high)
    )
    filled = float((volume["density"] > 0).mean())
    print(
        f"  map {grid.dims}, {grid.spacing[0]:.2f} A per sample, "
        f"{100 * filled:.1f}% above the contour"
    )

    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        kinds = client.actor_kinds()
        filters = client.filter_kinds()
        missing = {"surface", "volume"} - set(kinds)
        if missing:
            raise SystemExit(
                f"this build registers no {', '.join(sorted(missing))}; "
                f"it has {sorted(kinds)}."
            )
        if "cartoon" not in filters:
            raise SystemExit(
                f"this build registers no 'cartoon' filter; it has {sorted(filters)}"
            )

        root = client.create_object(f"{pdb_id} in EMD-{emd_id}")

        # The ribbon. No transform on either object: the two already agree in
        # angstroms, and moving one would be inventing a fit that the depositors
        # already did.
        ribbon = client.create_object(f"{pdb_id} cartoon")
        client.set_parent(ribbon, root)
        wanted = {name: structure[name] for name in ROLES.values() if name in structure}
        # Uploaded but not bound: these are what the actor panel's `colour`
        # picker offers, so the ribbon can be recoloured by chain or by B-factor
        # from the interface without re-running this script. An input only
        # offers arrays that are actually held.
        for extra in ("chain_index", "b_factor", "residue_index"):
            if extra in structure:
                wanted.setdefault(extra, structure[extra])
        held = client.upload_data(wanted)

        # The ribbon is a filter — atoms in, triangles out — and `surface` is
        # what makes those triangles opaque. There is no `mode` to set any more: an
        # absorbing ribbon is the same filter bound to `medium` instead, which is
        # the whole reason generating and displaying were split apart.
        #
        # Opaque is what this demo needs. The ribbon writes depth, so the
        # accumulation truncates every interval there and the map in front of it
        # dims it while the map behind does not.
        curve = client.add_filter(
            "cartoon",
            params={
                input_id: iris3d.Bind(held[name])
                for input_id, name in ROLES.items()
                if name in held
            },
        )
        # The loose arrays become one mesh, and that is what an actor binds.
        # Every actor bound to it shares the vertex buffers rather than
        # assembling its own copy.
        shape = client.add_filter(
            "geometry",
            params={
                "positions": iris3d.Bind(curve["positions"]),
                "indices": iris3d.Bind(curve["indices"]),
                "normals": iris3d.Bind(curve["normals"]),
            },
        )
        client.add_actor(
            "surface",
            parent=ribbon,
            params={
                "geometry": iris3d.Bind(shape["geometry"]),
                # No colours in the geometry, so the flat tint is what shows.
                "tint": (0.95, 0.75, 0.35),
            },
        )

        # The glycans, as SNFG symbols. 5A63 carries 20 sugars, and a cartoon
        # draws nothing for them — a sugar has no backbone — so the two actors
        # are complementary and share the same uploaded positions.
        if "residue_snfg" in structure and "glycan" in kinds:
            sugars = client.upload_data({"residue_snfg": structure["residue_snfg"]})
            client.add_actor(
                "glycan",
                parent=ribbon,
                params={
                    "positions": iris3d.Bind(held["positions"]),
                    "residue_index": iris3d.Bind(held["residue_index"]),
                    "residue_snfg": iris3d.Bind(sugars["residue_snfg"]),
                },
            )
            print(f"  {int((structure['residue_snfg'] > 0).sum())} sugars as SNFG symbols")

        # The map. `opacity` is the volume's absorbance per world unit, not the
        # contour level — the contour was applied when the map was loaded, and
        # the samples arrive normalised to 0..1 so this number means the same
        # thing from one entry to the next.
        cloud = client.create_object(f"EMD-{emd_id} density")
        client.set_parent(cloud, root)
        map_held = client.upload_data({"density": volume["density"]})
        client.add_actor(
            "volume",
            parent=cloud,
            params={
                "density": iris3d.Bind(map_held["density"]),
                # No `dims`: the array is (nx, ny, nz) and the actor reads the
                # grid's shape off it.
                "origin": grid.origin,
                "spacing": grid.spacing,
                # Absorbing well above emitting, which is the whole point here:
                # the map is meant to *block* the ribbon behind it, not to glow
                # over the top of it. Emission only far enough to give the
                # envelope some form of its own.
                "opacity": 12.0,
                "emission": 1.0,
                # Samples along the ray, not map resolution — the texture is
                # whatever was uploaded above, and no number here recovers
                # detail that was averaged away before it. What this does fix is
                # under-sampling: at 1.4 A per voxel across a ~150 A box a ray
                # crosses about 110 voxels, so 256 steps is only just over two
                # per voxel and thin features flicker. This is safe to raise —
                # the step length divides out of the integral, so the picture
                # holds still rather than getting brighter.
                "steps": 512.0,
                # A volume maps its own values, unlike every other kind: the
                # ramp is read per sample along the ray rather than once per
                # element, so materialising RGB per voxel would cost hundreds of
                # megabytes to save a texture fetch.
                "map": "cool-warm",
            },
        )

        print(
            f"loaded: {len(positions)} atoms as an opaque cartoon, "
            f"inside a {np.prod(grid.dims):,}-sample map"
        )


if __name__ == "__main__":
    main()
