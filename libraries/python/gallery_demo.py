"""Every filter kind and every actor kind, from one structure and its map.

Not a benchmark and not a tutorial: a scene where each thing the server can do
is on screen at once, so a change to any of them has somewhere obvious to be
looked at. The node view (top right) shows the whole graph, which is the other
reason it exists — more chain than a list reads well.

**One dataset for all of it.** Gamma-secretase: PDB 5A63 fitted into EMD-3061 at
3.4 A. It was chosen because a single entry exercises every kind that exists:

- 1245 residues over 10 chains, mixed helix, strand and coil — the cartoon;
- 20 glycan residues (NAG, BMA) — the SNFG symbols;
- two PC1 phosphatidylcholine lipids — ball and stick;
- per-atom ``b_factor`` and ``chain_index`` — the colour maps;
- the cryo-EM map itself — the volume, and a contour of it.

This replaced a gallery built from generated torii, an analytic orbital and a
benzene, plus two more structures fetched from RCSB. Real data throughout is
worth the loss: the shapes are the ones the renderer will actually meet, and a
kind that only ever ran against a clean synthetic case was not being tested by
being on screen.

What went with them is worth naming. The torus was a deliberately *closed*
mesh, which is what `medium` needs to integrate an interval correctly, and the
orbital was a field whose topology was known in advance. Both of those
assertions live in Rust unit tests — ``filter/contour.rs`` counts boundary
edges — so they are still made, just not here.

Laid out along x, one slot per kind, every slot binding the *same* uploaded
arrays. Nothing is uploaded twice: the readout at the top of the window is the
check on that.

Run against a started app::

    cargo run
    uv run python gallery_demo.py

Needs the network, for RCSB and EMDB both. The map is about 22 MB and is cached
between runs.
"""

from __future__ import annotations

import sys
import time

import iris3d
from iris3d import density, molecules

#: Gamma-secretase, and the map it was built into. See the module docstring for
#: why this pair rather than a spread of smaller ones.
PDB = "5a63"
EMD = "3061"

#: The second structure, and the reason there is one.
#:
#: 5A63 is cryo-EM and models no waters, so nothing there can prove that
#: selecting solvent *works* — only that asking for it correctly finds nothing.
#: Haemoglobin is X-ray at 1.74 A with 221 waters, four chains and a haem per
#: subunit, which is the smallest structure that exercises the other branch.
SOLVENT_PDB = "4hhb"

#: How far past the model to keep the map, in angstroms. Enough that the
#: density stands off the structure rather than being clipped tight to it.
MARGIN = 8.0

#: Gap between slots, on top of each piece's own width.
GAP = 30.0


def narrowed_by(client, held, picked, values: str) -> int:
    """One uploaded array, cut down to a selection, as a handle to bind.

    The whole of subsetting, now that an actor does none of it: a `subset`
    filter says which elements, and a `gather` per array applies it. Nothing is
    re-uploaded and nothing is copied on the client — the arrays being narrowed
    are the ones already on the server.
    """
    step = client.add_filter(
        "gather",
        params={
            "values": iris3d.Bind(held[values]),
            "indices": iris3d.Bind(picked["indices"]),
        },
    )
    return step["result"]


class Bench:
    """Places each piece in its own slot along x, sized to the model.

    Every slot holds the same molecule, so one width serves them all — unlike
    the old gallery, where a torus and a 64-sample grid needed measuring
    separately. The offset also re-centres: a PDB file's coordinates sit
    wherever the crystallographer left them, and 5A63's are nowhere near the
    origin.
    """

    def __init__(self, client: iris3d.Client, centre, width: float):
        self.client = client
        self.centre = centre
        self.width = width
        self.cursor = 0.0

    def place(self, name: str) -> int:
        handle = self.client.create_object(name)
        self.client.set_transform(
            handle,
            translation=(
                self.cursor - self.centre[0],
                -self.centre[1],
                -self.centre[2],
            ),
        )
        self.cursor += self.width + GAP
        return handle


def main() -> int:
    print(f"fetching {PDB} from RCSB...")
    structure = molecules.fetch(PDB, file_format="cif")
    positions = structure["positions"]
    low, high = positions.min(axis=0), positions.max(axis=0)
    centre = (low + high) / 2.0
    width = float((high - low)[0])

    print(f"fetching EMD-{EMD} from EMDB (about 22 MB, cached)...")
    # Cropped to the model plus a margin. The deposited box is nearly twice the
    # span of what is in it, so this is the difference between a few hundred
    # thousand samples and several million — and the padding would drag the
    # camera's framing out into empty space besides.
    #
    # No `sigma` and no `floor`: unset means the *deposited* contour level, the
    # one the people who built the model looked at it at, rather than a guess
    # from the histogram.
    volume, grid = density.fetch_emdb(EMD, factor=1, bounds=(tuple(low - MARGIN), tuple(high + MARGIN)))
    print(f"  map {grid.dims}, {grid.spacing[0]:.2f} A per sample")

    with iris3d.Client() as client:
        client.wait_until_ready()
        made: list[str] = []
        bench = Bench(client, centre, width)

        # --- one upload, bound by everything below -------------------------
        #
        # The whole structure goes up once. Every actor and filter after this
        # binds handles out of `atoms`, so drawing the same molecule six ways
        # costs one copy of the coordinates.
        atoms = client.upload_data(structure)
        field = client.upload_data({"density": volume["density"]})["density"]

        # --- cartoon -> colormap -> geometry -> surface, and the same mesh
        #     again as a medium ---------------------------------------------
        #
        # The deepest chain here, and the one the filter split was built for:
        # the ribbon is generated once, coloured as an ordinary array, and only
        # then assembled into vertices.
        ribbon = client.add_filter(
            "cartoon",
            params={
                "positions": iris3d.Bind(atoms["positions"]),
                "residue_index": iris3d.Bind(atoms["residue_index"]),
                "atom_name_index": iris3d.Bind(atoms["atom_name_index"]),
                "atom_name": iris3d.Bind(atoms["atom_name"]),
                "residue_sse": iris3d.Bind(atoms["residue_sse"]),
                "residue_chain_index": iris3d.Bind(atoms["residue_chain_index"]),
            },
        )
        # **Coloured by chain**, which is the ordinary thing to do to a multimer
        # and was unreachable until `gather` existed. The ribbon emits a residue
        # index per *vertex* and the chain is a property of the *residue*, so
        # the two levels have to be joined — `residue_chain_index[residue_index]`
        # — and nothing could evaluate that. `filter/cartoon.rs` has carried a
        # comment saying "or through a gather" since the day it was written.
        #
        # The map is `categorical`, not a ramp: chain 3 is not three fifths of
        # the way along anything, and a ramp would shift every colour whenever
        # the number of chains changed.
        chain_per_vertex = client.add_filter(
            "gather",
            params={
                "values": iris3d.Bind(atoms["residue_chain_index"]),
                "indices": iris3d.Bind(ribbon["residue_index"]),
            },
        )
        by_residue = client.add_filter(
            "colormap",
            params={
                "values": iris3d.Bind(chain_per_vertex["result"]),
                "map": "categorical",
            },
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
        # Both actors bind the *same* geometry handle. That is the whole point
        # of assembling once: drawing a thing two ways should add an actor and
        # no vertices, which the vertex count in the title bar confirms.
        client.add_actor(
            "surface",
            parent=bench.place("cartoon (surface)"),
            params={"geometry": iris3d.Bind(ribbon_mesh["geometry"])},
        )
        client.add_actor(
            "medium",
            parent=bench.place("cartoon (medium)"),
            params={"geometry": iris3d.Bind(ribbon_mesh["geometry"]), "absorbance": 8.0},
        )
        made += ["cartoon", "colormap", "geometry", "surface", "medium"]

        # --- points, coloured by B-factor ----------------------------------
        by_bfactor = client.add_filter(
            "colormap",
            params={"values": iris3d.Bind(atoms["b_factor"]), "map": "cool-warm"},
        )
        client.add_actor(
            "points",
            parent=bench.place("atoms (points)"),
            params={
                "positions": iris3d.Bind(atoms["positions"]),
                "colour": iris3d.Bind(by_bfactor["colour"]),
                "size": 0.4,
            },
        )
        made.append("points")

        # --- one chain, as its own cartoon ---------------------------------
        #
        # The case that needs `reindex`. Cutting atoms leaves `residue_index`
        # full of gaps — the surviving residues still carry their old numbers —
        # and every consumer downstream assumes it is dense. `reindex` closes
        # the gaps and reports which rows survived, and that `kept` array is
        # what narrows the residue-keyed arrays to match.
        #
        # Without it the ribbon would read secondary structure for residue 900
        # out of an array that now has 200 rows.
        first_chain = client.add_filter(
            "compare",
            params={
                "a": iris3d.Bind(atoms["chain_index"]),
                "op": "==",
                "value": 0.0,
            },
        )
        chain_atoms = client.add_filter(
            "subset", params={"mask": iris3d.Bind(first_chain["mask"])}
        )
        dense = client.add_filter(
            "reindex",
            params={"values": iris3d.Bind(narrowed_by(client, atoms, chain_atoms, "residue_index"))},
        )

        def per_residue(values: str) -> int:
            """A residue-keyed array, narrowed to the residues that survived."""
            step = client.add_filter(
                "gather",
                params={
                    "values": iris3d.Bind(atoms[values]),
                    "indices": iris3d.Bind(dense["kept"]),
                },
            )
            return step["result"]

        chain_ribbon = client.add_filter(
            "cartoon",
            params={
                "positions": iris3d.Bind(narrowed_by(client, atoms, chain_atoms, "positions")),
                "residue_index": iris3d.Bind(dense["result"]),
                "atom_name_index": iris3d.Bind(
                    narrowed_by(client, atoms, chain_atoms, "atom_name_index")
                ),
                "atom_name": iris3d.Bind(atoms["atom_name"]),
                "residue_sse": iris3d.Bind(per_residue("residue_sse")),
                "residue_chain_index": iris3d.Bind(per_residue("residue_chain_index")),
            },
        )
        # `logic` and `arithmetic`, on the way to a colour: flip the ribbon's
        # residue numbering end to end, so the chain reads C-to-N. Contrived as
        # a picture, honest as a test — it is the only place the two are wired
        # to something that has to come out right.
        flipped = client.add_filter(
            "arithmetic",
            params={
                "a": iris3d.Bind(chain_ribbon["residue_index"]),
                "op": "*",
                "value": -1.0,
            },
        )
        chain_colour = client.add_filter(
            "colormap", params={"values": iris3d.Bind(flipped["result"])}
        )
        chain_mesh = client.add_filter(
            "geometry",
            params={
                "positions": iris3d.Bind(chain_ribbon["positions"]),
                "indices": iris3d.Bind(chain_ribbon["indices"]),
                "normals": iris3d.Bind(chain_ribbon["normals"]),
                "colour": iris3d.Bind(chain_colour["colour"]),
            },
        )
        client.add_actor(
            "surface",
            parent=bench.place("chain A (cartoon)"),
            params={"geometry": iris3d.Bind(chain_mesh["geometry"])},
        )
        made += ["compare", "reindex", "arithmetic"]

        # --- ball and stick, on the lipids ---------------------------------
        #
        # **The selection is nodes, not numpy.** Nothing extra is uploaded: the
        # lipids are picked out of the arrays already on the server, by a rule
        # the server can be asked to change. Dragging `match` from PC1 to NAG
        # redraws this as the sugars without a byte crossing the wire.
        #
        # Six nodes, and each one does exactly one thing:
        #
        #   match     names the residues wanted, per *residue*
        #   gather    reads that mask through residue_index, giving per *atom*
        #   subset    turns the mask into the indices it keeps
        #   gather    narrows the positions      \  what the actor binds,
        #   gather    narrows the elements        > already narrowed, so the
        #   renumber  narrows and rewires bonds  /   actor decides nothing
        lipid_residues = client.add_filter(
            "match",
            params={
                "index": iris3d.Bind(atoms["residue_name_index"]),
                "values": iris3d.Bind(atoms["residue_name"]),
                "text": "PC1",
            },
        )
        # A mask per residue becomes a mask per atom by reading it through the
        # atom's own residue index. This is the hierarchy join, and it is the
        # same `gather` that colours a ribbon by chain.
        lipid_atoms = client.add_filter(
            "gather",
            params={
                "values": iris3d.Bind(lipid_residues["mask"]),
                "indices": iris3d.Bind(atoms["residue_index"]),
            },
        )
        picked = client.add_filter(
            "subset", params={"mask": iris3d.Bind(lipid_atoms["result"])}
        )

        # Bonds are the one array a gather cannot narrow: they name atoms by
        # index, so cutting atoms means dropping any bond with a cut end *and*
        # renumbering the survivors into the space the kept atoms now occupy.
        lipid_bonds = client.add_filter(
            "renumber",
            params={
                "connectivity": iris3d.Bind(atoms["bonds"]),
                "indices": iris3d.Bind(picked["indices"]),
            },
        )
        client.add_actor(
            "ball-and-stick",
            parent=bench.place("lipids (ball and stick)"),
            params={
                "positions": iris3d.Bind(narrowed_by(client, atoms, picked, "positions")),
                "elements": iris3d.Bind(narrowed_by(client, atoms, picked, "elements")),
                "bonds": iris3d.Bind(lipid_bonds["connectivity"]),
                "atom_scale": 0.3,
            },
        )
        made += ["ball-and-stick", "match", "gather", "subset", "renumber"]

        # --- glycans --------------------------------------------------------
        #
        # A second way of drawing the same atoms rather than a different set of
        # them: the SNFG symbols are placed per *residue*, and the filter reads
        # which residues are sugars off `residue_snfg`. No subsetting needed,
        # which is why this one was already clean.
        #
        # `colour` comes out of the filter rather than a `colormap`, because
        # SNFG's palette identifies the sugar and is not a choice to offer.
        symbols = client.add_filter(
            "glycan",
            params={
                "positions": iris3d.Bind(atoms["positions"]),
                "residue_index": iris3d.Bind(atoms["residue_index"]),
                "residue_snfg": iris3d.Bind(atoms["residue_snfg"]),
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
            parent=bench.place("glycans (SNFG)"),
            params={"geometry": iris3d.Bind(snfg["geometry"])},
        )
        made.append("glycan")

        # --- the map, raymarched and contoured ------------------------------
        #
        # One upload feeding both, which is the point of the grid coming off the
        # array's own shape: `volume` and `contour` read it the same way, so
        # neither needs telling what it can already see.
        client.add_actor(
            "volume",
            parent=bench.place(f"EMD-{EMD} (volume)"),
            params={
                "density": iris3d.Bind(field),
                "origin": grid.origin,
                "spacing": grid.spacing,
                "opacity": 12.0,
                "emission": 1.0,
                "steps": 256.0,
                "map": "cool-warm",
            },
        )
        shell = client.add_filter(
            "contour",
            params={
                "field": iris3d.Bind(field),
                "colour_field": iris3d.Bind(field),
                "level": 0.25,
                "origin": grid.origin,
                "spacing": grid.spacing,
            },
        )
        client.add_actor(
            "surface",
            parent=bench.place(f"EMD-{EMD} (isosurface)"),
            params={"geometry": iris3d.Bind(shell["geometry"])},
        )
        made += ["volume", "contour"]

        # --- the solvent pair, on a structure that actually has solvent -----
        #
        # 5A63 is cryo-EM at 3.4 A and models no waters at all, so the whole
        # water path there can only ever prove the *negative* case — `match`
        # correctly reporting that it found no HOH. That is worth having and it
        # is not a test.
        #
        # 4HHB is the complement, and deliberately so: X-ray at 1.74 A, four
        # chains of real quaternary structure, 221 modelled waters and a haem in
        # each subunit. Between the two, every branch of the selection code has
        # a structure that exercises it.
        print(f"fetching {SOLVENT_PDB} for the solvent pair...")
        second = molecules.fetch(SOLVENT_PDB, file_format="cif")
        low2, high2 = second["positions"].min(axis=0), second["positions"].max(axis=0)
        bench.centre = (low2 + high2) / 2.0
        bench.width = float((high2 - low2)[0])
        held = client.upload_data(second)

        waters = client.add_filter(
            "match",
            params={
                "index": iris3d.Bind(held["residue_name_index"]),
                "values": iris3d.Bind(held["residue_name"]),
                "text": "HOH",
            },
        )
        # Per residue to per atom. A water is one residue of three atoms, so the
        # two counts genuinely differ and the join is doing real work.
        water_atoms = client.add_filter(
            "gather",
            params={
                "values": iris3d.Bind(waters["mask"]),
                "indices": iris3d.Bind(held["residue_index"]),
            },
        )
        just_water = client.add_filter(
            "subset", params={"mask": iris3d.Bind(water_atoms["result"])}
        )
        client.add_actor(
            "points",
            parent=bench.place(f"{SOLVENT_PDB.upper()} waters (points)"),
            params={
                "positions": iris3d.Bind(narrowed_by(client, held, just_water, "positions")),
                "size": 0.5,
            },
        )

        # And the other half of the same mask, which is what anyone actually
        # wants: the structure with the solvent taken off it. `logic` earns its
        # place here — "not water" cannot be a comparison, because the thing
        # being negated came from matching text.
        solute = client.add_filter(
            "logic", params={"a": iris3d.Bind(water_atoms["result"]), "op": "not"}
        )
        dry = client.add_filter("subset", params={"mask": iris3d.Bind(solute["mask"])})
        dry_bonds = client.add_filter(
            "renumber",
            params={
                "connectivity": iris3d.Bind(held["bonds"]),
                "indices": iris3d.Bind(dry["indices"]),
            },
        )
        client.add_actor(
            "ball-and-stick",
            parent=bench.place(f"{SOLVENT_PDB.upper()} without solvent"),
            params={
                "positions": iris3d.Bind(narrowed_by(client, held, dry, "positions")),
                "elements": iris3d.Bind(narrowed_by(client, held, dry, "elements")),
                "bonds": iris3d.Bind(dry_bonds["connectivity"]),
                "atom_scale": 0.25,
            },
        )
        made.append("logic")

        # The point of the secondary structure: this match must *hit*. A
        # complaint here means the water path is only ever being proved by its
        # negative case, which is how it looked before 4HHB was added.
        time.sleep(2.0)
        for entry in client.list_filters():
            if entry.handle == waters.handle and entry.problem:
                raise SystemExit(
                    f"the solvent match found nothing in {SOLVENT_PDB}: {entry.problem}"
                )

        print()
        print("filters:", len(client.list_filters()))
        print("objects:", len(client.list_objects()))
        kinds = set(client.actor_kinds()) | set(client.filter_kinds())
        missing = sorted(kinds - set(made))
        print("covered:", ", ".join(sorted(set(made))))
        print("missing:", ", ".join(missing) if missing else "nothing")
        return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
