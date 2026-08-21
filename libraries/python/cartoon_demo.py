"""Loads two structures into a running iris3d and draws them as cartoons.

Two on purpose, because they exercise different halves of the ribbon code:

- **1UBQ**, ubiquitin. One chain, mixed alpha and beta, and the beta sheet ends
  in strands that should each carry an arrowhead. This is the structure to judge
  the renderer by — if the arrows point the wrong way or the sheet ripples, it
  shows here.
- **1BNA**, the Drew-Dickerson B-DNA dodecamer. No protein and no secondary
  structure at all, so it goes down the nucleic path: a different trace atom, a
  different direction atom, and the swapped frame.

Both are downloaded from RCSB, so this needs the network. Run it against an
already-running app:

    uv run python cartoon_demo.py

Colouring is by ``residue_index``, which gives the N-to-C rainbow every viewer
uses. That is a deliberate choice of this script and not a default: the cartoon
actor colours per residue, and the residue index is simply the most legible
thing to key on.
"""

from __future__ import annotations

import sys

import iris3d
from iris3d import molecules

#: What to load when nothing is named on the command line. The offsets are
#: measured below rather than guessed, so this is only the order they appear in.
#: Pass entry ids as arguments to load something else::
#:
#:     uv run python cartoon_demo.py 1kx5
DEFAULT_ENTRIES = ["1ubq", "1bna"]

#: The six arrays the cartoon filter binds, as input id -> the name `molecules`
#: gives it.
#:
#: They match one for one here, which is not luck: `molecules.arrays_from_atoms`
#: names them after biotite's annotation categories, and the actor's inputs were
#: named after the same thing. A loader that called them something else would
#: change this mapping and nothing else.
ROLES = {
    "positions": "positions",
    "residue_index": "residue_index",
    "atom_name_index": "atom_name_index",
    "atom_name": "atom_name",
    "residue_sse": "residue_sse",
    "residue_chain_index": "residue_chain_index",
}


def place(arrays, cursor, gap=8.0):
    """Puts each structure in its own slot along x, sized to its own extent."""
    positions = arrays["positions"]
    low, high = positions.min(axis=0), positions.max(axis=0)
    width = float(high[0] - low[0])
    centre = float(low[0] + high[0]) / 2.0
    # Shift so the structure's own centre lands in the middle of its slot, and
    # drop it onto the origin in y and z as well — a PDB file's coordinates sit
    # wherever the crystallographer left them.
    offset = (
        cursor + width / 2.0 - centre,
        -float(low[1] + high[1]) / 2.0,
        -float(low[2] + high[2]) / 2.0,
    )
    return offset, cursor + width + gap


def main():
    entries = sys.argv[1:] or DEFAULT_ENTRIES

    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        print("connected")

        # Ask rather than assume, on both sides: the ribbon is a filter kind and
        # what draws it is an actor kind, and a hardcoded list of either would
        # eventually name something that silently does nothing.
        available = client.actor_kinds()
        filters = client.filter_kinds()
        if "cartoon" not in filters:
            raise SystemExit(
                f"this build registers no 'cartoon' filter; it has "
                f"{sorted(filters)}"
            )
        if "surface" not in available:
            raise SystemExit(
                f"this build registers no 'surface' kind; it has {sorted(available)}"
            )

        root = client.create_object("cartoons")
        cursor = 0.0

        for entry in entries:
            print(f"fetching {entry} from RCSB...")
            # mmCIF rather than PDB. Every entry has one, including the large
            # assemblies that outgrew the PDB format's chain-id and atom-count
            # limits, so this is the format that does not need a special case.
            arrays = molecules.fetch(entry, file_format="cif")

            handle = client.create_object(entry)
            client.set_parent(handle, root)
            offset, cursor = place(arrays, cursor)
            client.set_transform(handle, translation=offset)

            wanted = {name: arrays[name] for name in ROLES.values() if name in arrays}
            missing = set(ROLES.values()) - set(wanted)
            if missing:
                # Not fatal: `residue_sse` is absent whenever P-SEA could not
                # assign anything, which is exactly the case for 1BNA. Say so
                # rather than letting a plain tube look like a bug.
                print(f"  {entry}: no {', '.join(sorted(missing))}")

            # Uploaded but not bound: these are what the actor panel's `colour`
            # picker offers, so the ribbon can be recoloured by chain or by
            # B-factor from the interface. An input only offers arrays that are
            # actually held, so leaving them out makes the control look broken.
            for extra in ("chain_index", "b_factor"):
                if extra in arrays:
                    wanted.setdefault(extra, arrays[extra])

            held = client.upload_data(wanted)

            # The ribbon is a *filter*: atoms in, triangles out, drawing nothing.
            # What draws it is a separate choice, which is the whole point —
            # `surface` for a lit ribbon, `medium` for one you see through, and
            # the curve is solved once either way.
            ribbon = client.add_filter(
                "cartoon",
                params={
                    input_id: iris3d.Bind(held[name])
                    for input_id, name in ROLES.items()
                    if name in held
                },
            )

            # The N-to-C rainbow. The cartoon emits a residue index per *vertex*,
            # so colouring it is an ordinary colour map over an ordinary array —
            # no cartoon-specific colour path anywhere.
            rainbow = client.add_filter(
                "colormap",
                params={
                    "values": iris3d.Bind(ribbon["residue_index"]),
                    "map": "viridis",
                },
            )

            # The arrays become one mesh here, and that mesh is what an actor
            # binds. Every actor bound to it references the same vertex buffers,
            # so drawing this ribbon as `surface` *and* as `medium` is one upload
            # rather than two — add the second actor with the same handle to
            # see it. The colours have to be part of the assembly for the same
            # reason: a shared buffer cannot be painted per consumer.
            shape = client.add_filter(
                "geometry",
                params={
                    "positions": iris3d.Bind(ribbon["positions"]),
                    "indices": iris3d.Bind(ribbon["indices"]),
                    "normals": iris3d.Bind(ribbon["normals"]),
                    "colour": iris3d.Bind(rainbow["colour"]),
                },
            )

            client.add_actor(
                "surface",
                parent=handle,
                params={"geometry": iris3d.Bind(shape["geometry"])},
            )
            # Glycans, as a second chain under the same object. A cartoon draws
            # nothing for a sugar — it has no backbone — so the two are
            # complementary rather than overlapping, and they share the same
            # uploaded positions.
            #
            # Same shape as the ribbon above, minus the colour map: SNFG's
            # palette is the notation, so the filter hands out `colour` itself
            # rather than a scalar for something downstream to interpret.
            if "residue_snfg" in arrays and "glycan" in filters:
                sugars = client.upload_data({"residue_snfg": arrays["residue_snfg"]})
                symbols = client.add_filter(
                    "glycan",
                    params={
                        "positions": iris3d.Bind(held["positions"]),
                        "residue_index": iris3d.Bind(held["residue_index"]),
                        "residue_snfg": iris3d.Bind(sugars["residue_snfg"]),
                    },
                )
                snfg = client.add_filter(
                    "geometry",
                    params={
                        "positions": iris3d.Bind(symbols["positions"]),
                        "indices": iris3d.Bind(symbols["indices"]),
                        "normals": iris3d.Bind(symbols["normals"]),
                        "colour": iris3d.Bind(symbols["colour"]),
                    },
                )
                client.add_actor(
                    "surface",
                    parent=handle,
                    params={"geometry": iris3d.Bind(snfg["geometry"])},
                )
                count = int((arrays["residue_snfg"] > 0).sum())
                print(f"  {entry}: {count} sugar residues as SNFG symbols")

            residues = int(arrays["residue_index"].max()) + 1
            print(f"  {entry}: {len(arrays['positions'])} atoms, {residues} residues")

        print("done")


if __name__ == "__main__":
    main()
