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

#: The six arrays a cartoon binds, as input id -> the name `molecules` gives it.
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

        # Ask rather than assume. Which kinds exist is decided by the server,
        # and a hardcoded list here would eventually name something that
        # silently does nothing.
        available = client.actor_kinds()
        if "cartoon" not in available:
            raise SystemExit(
                f"this build registers no 'cartoon' kind; it has "
                f"{sorted(available)}"
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
            params = {
                input_id: iris3d.Bind(held[name])
                for input_id, name in ROLES.items()
                if name in held
            }
            # The N-to-C rainbow. `residue_index` is per atom; the actor reduces
            # it to one value per residue by taking each residue's first atom.
            params["colour"] = iris3d.Bind(held["residue_index"])

            client.add_actor(
                "cartoon",
                parent=handle,
                params=params,
                coloring=iris3d.Coloring(map="viridis"),
            )
            # Glycans, as a second actor under the same object. A cartoon draws
            # nothing for a sugar — it has no backbone — so the two are
            # complementary rather than overlapping, and they share the same
            # uploaded positions.
            if "residue_snfg" in arrays and "glycan" in available:
                sugars = client.upload_data({"residue_snfg": arrays["residue_snfg"]})
                client.add_actor(
                    "glycan",
                    parent=handle,
                    params={
                        "positions": iris3d.Bind(held["positions"]),
                        "residue_index": iris3d.Bind(held["residue_index"]),
                        "residue_snfg": iris3d.Bind(sugars["residue_snfg"]),
                    },
                )
                count = int((arrays["residue_snfg"] > 0).sum())
                print(f"  {entry}: {count} sugar residues as SNFG symbols")

            residues = int(arrays["residue_index"].max()) + 1
            print(f"  {entry}: {len(arrays['positions'])} atoms, {residues} residues")

        print("done")


if __name__ == "__main__":
    main()
