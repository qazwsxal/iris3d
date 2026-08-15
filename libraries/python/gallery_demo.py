"""Every filter kind and every actor kind, in one scene.

Not a benchmark and not a tutorial: a scene where each thing the server can do
is on screen at once, so a change to any of them has somewhere obvious to be
looked at. The node view (top right) shows the whole graph, which is the other
reason it exists — nine actors and six filters is more chain than a list reads
well.

Laid out along x so nothing overlaps, since each piece is its own object with
its own transform.

Run against a started app::

    cargo run
    uv run python gallery_demo.py

Needs the network for the two molecular pieces: crambin for the cartoon and a
broadly neutralising antibody complex for the glycans. Everything else is
generated here.
"""

from __future__ import annotations

import sys

import numpy as np

import iris3d
from iris3d import molecules, testdata

#: Small and fast to fetch: 46 residues, one chain, a real mix of helix and
#: sheet so a cartoon has something to show.
CARTOON_PDB = "1crn"

#: Heavily glycosylated, which is the point — a structure with no sugars leaves
#: `residue_snfg` off the wire entirely and the glycan actor draws nothing.
GLYCAN_PDB = "5fyl"


def place(client: iris3d.Client, name: str, x: float) -> int:
    """An object at its own spot along x, so the pieces do not sit inside one
    another."""
    handle = client.create_object(name)
    client.set_transform(handle, translation=(x, 0.0, 0.0))
    return handle


def main() -> int:
    with iris3d.Client() as client:
        client.wait_until_ready()
        made: list[str] = []

        # --- geometry -> surface, and the same mesh as a medium ------------
        #
        # Both actors bind the *same* geometry handle. That is the whole point
        # of assembling once: drawing a thing two ways should add an actor and
        # no vertices, which the readout at the top of the window confirms.
        torus = testdata.torus_mesh()
        held = client.upload_data(
            {
                "torus_positions": torus["positions"],
                "torus_indices": torus["indices"],
                "torus_height": torus["positions"][:, 1].copy(),
            }
        )
        tint = client.add_filter(
            "colormap", params={"values": iris3d.Bind(held["torus_height"])}
        )
        shape = client.add_filter(
            "geometry",
            params={
                "positions": iris3d.Bind(held["torus_positions"]),
                "indices": iris3d.Bind(held["torus_indices"]),
                "colour": iris3d.Bind(tint["colour"]),
            },
        )
        client.add_actor(
            "surface",
            parent=place(client, "torus (surface)", -30.0),
            params={"geometry": iris3d.Bind(shape["geometry"])},
        )
        client.add_actor(
            "medium",
            parent=place(client, "torus (medium)", -15.0),
            params={"geometry": iris3d.Bind(shape["geometry"]), "absorbance": 8.0},
        )
        made += ["colormap", "geometry", "surface", "medium"]

        # --- points -------------------------------------------------------
        cloud = testdata.torus_points()
        points = client.upload_data(
            {"cloud_positions": cloud["positions"], "cloud_angle": cloud["angle"]}
        )
        cloud_colour = client.add_filter(
            "colormap",
            params={"values": iris3d.Bind(points["cloud_angle"]), "map": "cool-warm"},
        )
        client.add_actor(
            "points",
            parent=place(client, "point cloud", 0.0),
            params={
                "positions": iris3d.Bind(points["cloud_positions"]),
                "colour": iris3d.Bind(cloud_colour["colour"]),
                "size": 0.08,
            },
        )
        made.append("points")

        # --- volume, and a contour of the same field ----------------------
        #
        # One upload, drawn two ways: raymarched as a medium, and extracted as
        # a surface. `contour` reads the grid off the array's own shape, so the
        # field goes up as (n, n, n) rather than ravelled.
        orbital, grid = testdata.hydrogen_orbital(n=48)
        probability = orbital["probability"].reshape(grid.dims)

        # One upload feeding both, which is the point of the grid coming off the
        # array's own shape: `volume` and `contour` read it the same way, so
        # neither needs telling what it can already see.
        field = client.upload_data({"orbital": probability})["orbital"]
        client.add_actor(
            "volume",
            parent=place(client, "orbital (volume)", 20.0),
            params={
                "density": iris3d.Bind(field),
                "origin": grid.origin,
                "spacing": grid.spacing,
                "opacity": 0.4,
            },
        )
        surface = client.add_filter(
            "contour",
            params={
                "field": iris3d.Bind(field),
                "colour_field": iris3d.Bind(field),
                "level": 0.35,
                "origin": grid.origin,
                "spacing": grid.spacing,
            },
        )
        client.add_actor(
            "surface",
            parent=place(client, "orbital (isosurface)", 45.0),
            params={"geometry": iris3d.Bind(surface["geometry"])},
        )
        made += ["volume", "contour"]

        # --- ball and stick -----------------------------------------------
        benzene = testdata.benzene()
        molecule = client.upload_data(
            {
                "benzene_positions": benzene["positions"],
                "benzene_elements": benzene["elements"],
                "benzene_bonds": benzene["bonds"],
            }
        )
        client.add_actor(
            "ball-and-stick",
            parent=place(client, "benzene", 60.0),
            params={
                "positions": iris3d.Bind(molecule["benzene_positions"]),
                "elements": iris3d.Bind(molecule["benzene_elements"]),
                "bonds": iris3d.Bind(molecule["benzene_bonds"]),
            },
        )
        made.append("ball-and-stick")

        # --- cartoon -> colormap -> geometry -> surface --------------------
        #
        # The deepest chain here, and the one the split was built for: the
        # ribbon is generated once, coloured as an ordinary array, and only then
        # assembled into vertices.
        print(f"fetching {CARTOON_PDB} for the cartoon...")
        protein = molecules.fetch(CARTOON_PDB)
        atoms = client.upload_data(
            {f"crambin_{name}": values for name, values in protein.items()}
        )
        ribbon = client.add_filter(
            "cartoon",
            params={
                "positions": iris3d.Bind(atoms["crambin_positions"]),
                "residue_index": iris3d.Bind(atoms["crambin_residue_index"]),
                "atom_name_index": iris3d.Bind(atoms["crambin_atom_name_index"]),
                "atom_name": iris3d.Bind(atoms["crambin_atom_name"]),
                "residue_sse": iris3d.Bind(atoms["crambin_residue_sse"]),
                "residue_chain_index": iris3d.Bind(
                    atoms["crambin_residue_chain_index"]
                ),
            },
        )
        by_residue = client.add_filter(
            "colormap", params={"values": iris3d.Bind(ribbon["residue_index"])}
        )
        ribbon_mesh = client.add_filter(
            "geometry",
            params={
                "positions": iris3d.Bind(ribbon["positions"]),
                "indices": iris3d.Bind(ribbon["indices"]),
                "normals": iris3d.Bind(ribbon["normals"]),
                "colour": iris3d.Bind(by_residue["colour"]),
            },
        )
        client.add_actor(
            "surface",
            parent=place(client, "crambin (cartoon)", 80.0),
            params={"geometry": iris3d.Bind(ribbon_mesh["geometry"])},
        )
        made.append("cartoon")

        # --- glycan --------------------------------------------------------
        print(f"fetching {GLYCAN_PDB} for the glycans...")
        glyco = molecules.fetch(GLYCAN_PDB)
        if "residue_snfg" not in glyco:
            print(f"!! {GLYCAN_PDB} carries no sugars; skipping the glycan actor")
        else:
            sugars = client.upload_data(
                {
                    "glyco_positions": glyco["positions"],
                    "glyco_residue_index": glyco["residue_index"],
                    "glyco_residue_snfg": glyco["residue_snfg"],
                }
            )
            client.add_actor(
                "glycan",
                parent=place(client, "glycans (SNFG)", 110.0),
                params={
                    "positions": iris3d.Bind(sugars["glyco_positions"]),
                    "residue_index": iris3d.Bind(sugars["glyco_residue_index"]),
                    "residue_snfg": iris3d.Bind(sugars["glyco_residue_snfg"]),
                },
            )
            made.append("glycan")

        print()
        print("filters:", len(client.list_filters()))
        print("objects:", len(client.list_objects()))
        kinds = set(client.actor_kinds()) | set(client.filter_kinds())
        missing = sorted(kinds - set(made))
        print("covered:", ", ".join(sorted(set(made))))
        print("missing:", ", ".join(missing) if missing else "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
