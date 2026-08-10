"""Scratch client: loads the sample datasets into a running iris3d."""

import os

import iris3d
from iris3d import molecules, testdata

def layout(datasets, gap=3.0):
    """Places each dataset in its own slot along x, sized to its own extent.

    Measured rather than hardcoded because the samples differ wildly in scale
    — a torus spans 8 units, heme about 20 angstroms — and fixed offsets mean
    one sample quietly swallowing the next.
    """
    placements = {}
    cursor = 0.0
    for name, arrays in datasets.items():
        positions = arrays.get("positions")
        if positions is None or len(positions) == 0:
            placements[name] = (cursor, 0.0, 0.0)
            continue
        low, high = positions.min(axis=0), positions.max(axis=0)
        width = float(high[0] - low[0])
        centre = float(low[0] + high[0]) / 2.0
        # Shift so this dataset's own centre lands in the middle of its slot.
        placements[name] = (cursor + width / 2.0 - centre, 0.0, 0.0)
        cursor += width + gap
    # The cursor comes back too, so a dataset that is not in this mapping can
    # still be given the next free slot.
    return placements, cursor


def datasets():
    """The analytic samples, plus real structures read through biotite.

    Heme comes from the Chemical Component Dictionary that biotite ships, so
    it needs no network access — real geometry, real bonds, a coordinated
    iron, and aromatic rings that exercise the bond-type encoding.

    Set ``IRIS3D_FETCH_PDB`` to an entry id (say ``1ubq``) to also download
    that structure from RCSB. Off by default so a normal run stays offline.
    """
    everything = testdata.examples()
    everything["heme (biotite)"] = molecules.residue("HEM")

    entry = os.environ.get("IRIS3D_FETCH_PDB")
    if entry:
        print(f"fetching {entry} from RCSB...")
        everything[f"{entry} (rcsb)"] = molecules.fetch(entry)
    return everything


def hydrogen(client, root, cursor, gap=3.0):
    """Adds the 3d_z2 orbital as a grid, drawn as a volume.

    Kept out of :func:`datasets` because a grid does not fit a plain
    ``{name: arrays}`` mapping. It is *declared*, not inferred: a grid's sample
    positions are implicit, so no array reveals it and none carries the spacing.
    That is also why it is the only sample here that uploads no positions at all
    — 64³ samples state their geometry in nine numbers.

    Watch for two lobes along z, a ring around the waist, and a gap between
    them. Lobes lying sideways would mean the axis order is read the wrong way
    round.
    """
    arrays, grid = testdata.hydrogen_orbital(n=64)
    handle = client.upload_object("hydrogen 3dz2", arrays, grid=grid)
    client.set_parent(handle, root)

    # The grid is centred on its own origin, so only the slot offset applies.
    width = grid.dims[0] * grid.spacing[0]
    client.set_transform(handle, translation=(cursor + width / 2.0, 0.0, 0.0))

    # The upload draws nothing on its own, so the volume is asked for outright.
    # Settings go in at the same time — before, this took a second call to find
    # the actor the upload had quietly made and a third to configure it.
    #
    # Two separate choices here, which is the whole point of the controls:
    # `density` says what makes the volume solid, and the colouring says what
    # tints it. Opacity is turned well up because most of the box is nearly
    # empty, and at 1.0 the lobes barely register.
    #
    # Density is the probability, the square of the amplitude, so it has no
    # sign to lose. Colour is the signed amplitude on a diverging map, which is
    # what shows the lobes and the ring as opposite phases rather than as one
    # undifferentiated cloud. Set both to "probability" to see the difference.
    #
    # This grid spans 24 units against the torus's 8, so it dominates the
    # default framing. Orbit round it rather than judging it from where the
    # camera lands.
    client.add_actor(
        handle,
        "volume",
        params={
            "density": "probability",
            "mode": "blend",
            "opacity": 12.0,
            "steps": 256.0,
        },
        coloring=iris3d.Coloring(field="amplitude", map="cool-warm"),
    )
    return handle


def bind(client, kind, arrays):
    """Uploads what a kind needs and returns the bindings for it.

    The mapping from an array's name to the input it feeds is *this script's*,
    not the server's. Nothing infers a role from a name any more: these samples
    happen to call their coordinates "positions", and a sample that called them
    "xyz" would bind exactly the same way with one line changed here.
    """
    # Input id -> the name this script's data happens to use for it.
    roles = {
        "points": {"positions": "positions"},
        "surface": {"positions": "positions", "indices": "indices", "normals": "normals"},
        "ball-and-stick": {
            "positions": "positions",
            "elements": "elements",
            "bonds": "bonds",
        },
    }[kind]
    wanted = {name: arrays[name] for name in roles.values() if name in arrays}

    # Whichever scalar the sample carries, if any, to colour by. B-factors for a
    # fetched structure; the analytic samples carry their own.
    scalar = next(
        (name for name in ("b_factor", "von_mises", "height") if name in arrays),
        None,
    )
    if scalar is not None:
        wanted[scalar] = arrays[scalar]

    held = client.upload_data(wanted)
    params = {
        input_id: iris3d.Bind(held[name])
        for input_id, name in roles.items()
        if name in held
    }
    if scalar is not None:
        params["colour"] = iris3d.Bind(held[scalar])
    return params


def main():
    # Waits for the app to come up, so this can be launched alongside it.
    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        print("connected")

        # Everything hangs off one empty object, so the whole sample set can be
        # moved — or removed — as a unit.
        root = client.create_object("examples")

        everything = datasets()
        placements, cursor = layout(everything)
        for name, arrays in everything.items():
            handle = client.upload_object(name, arrays)
            client.set_parent(handle, root)
            client.set_transform(handle, translation=placements[name])

        hydrogen(client, root, cursor)

        # An upload puts data in the scene; it does not decide how the data
        # looks. The server reports what it can draw and which datasets each
        # kind accepts, and choosing among them is this script's business. First
        # kind that fits is a fine policy for a sample loader — a real client
        # would offer the list.
        preferred: dict[str, str] = {}
        for kind in client.actor_kinds().values():
            for dataset in kind.supports:
                preferred.setdefault(dataset, kind.id)
        for summary in client.list_objects():
            wanted = preferred.get(summary.dataset_kind)
            # The grid already has the volume `hydrogen` asked for, and the root
            # is an empty grouping node that nothing can draw.
            if wanted is None or summary.actors:
                continue
            # These read bound arrays rather than the object's dataset, so their
            # data goes in through `upload_data` and is named at the actor. Only
            # `volume` still takes its own from the object it draws.
            if wanted in ("points", "surface", "ball-and-stick"):
                client.add_actor(
                    summary.handle, wanted, params=bind(client, wanted, everything[summary.name])
                )
            else:
                client.add_actor(summary.handle, wanted)

        print(f"\n{'handle':<8}{'object':<18}{'kind':<11}{'drawn as':<16}arrays")
        print("-" * 78)
        for summary in sorted(client.list_objects(), key=lambda s: s.handle):
            indent = "  " if summary.parent is not None else ""
            arrays = ", ".join(f"{b.name}{list(b.shape)}" for b in summary.buffers)
            print(
                f"{summary.handle:<8}{indent + summary.name:<18}"
                f"{summary.dataset_kind:<11}"
                f"{', '.join(r.kind for r in summary.actors) or '-':<16}{arrays}"
            )

        total = sum(s.total_bytes for s in client.list_objects())
        print(f"\n{total / 1024:.1f} KiB resident")


if __name__ == "__main__":
    main()
