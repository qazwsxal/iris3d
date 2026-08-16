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
    # A grouping node with no data of its own: the arrays go in separately and
    # the actor binds them, so the object is only a place in the tree.
    handle = client.create_object("hydrogen 3dz2")
    client.set_parent(handle, root)

    # The grid is centred on its own origin, so only the slot offset applies.
    width = grid.dims[0] * grid.spacing[0]
    client.set_transform(handle, translation=(cursor + width / 2.0, 0.0, 0.0))

    # Reshaped to (nx, ny, nz) on the way up. A volume input declares
    # `[0, 0, 0]` and takes the grid's extent from the array, so a ravelled
    # field is refused at bind time rather than drawn wrongly.
    held = client.upload_data(
        {
            "probability": arrays["probability"].reshape(grid.dims),
            "amplitude": arrays["amplitude"].reshape(grid.dims),
        }
    )

    # The arrangement of the samples travels as three vectors rather than as an
    # array, which is the whole reason a grid is worth having: 64³ samples state
    # their geometry in nine numbers instead of 262144 coordinates.
    #
    # Two separate choices here, which is the whole point of the controls:
    # `density` says what makes the volume solid, and the colouring says what
    # tints it. Opacity is turned well up because most of the box is nearly
    # empty, and at 1.0 the lobes barely register.
    #
    # There is no `mode` any more. The volume kind used to belong to a
    # standard-pipeline backend that alpha-blended and offered a choice of how;
    # this one deposits absorbance into the moment buffer, which is the only
    # thing it does. An unknown parameter is dropped silently rather than
    # refused, so passing the old one looked like it worked.
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
        "volume",
        parent=handle,
        params={
            "density": iris3d.Bind(held["probability"]),
            "colour": iris3d.Bind(held["amplitude"]),
            # No `dims`: the array is (nx, ny, nz) and the actor reads the
            # grid's shape off it.
            "origin": grid.origin,
            "spacing": grid.spacing,
            "opacity": 12.0,
            # The orbital is the one sample that should glow rather than only
            # absorb: it is a probability cloud, not a solid, and with emission
            # at the default of 1 the lobes read as smoke instead of light.
            "emission": 4.0,
            "steps": 256.0,
            # A volume maps its own values; see `GridStyle::map`.
            "map": "cool-warm",
        },
    )
    return handle


def kind_for(arrays):
    """This script's choice of representation for its own data.

    Inference, but on the right side of the wire. These arrays were named by
    ``testdata`` and ``molecules`` a few lines away, so recognising them here is
    reading one's own notes — whereas the server doing it meant guessing from
    names it had never seen before.
    """
    if "elements" in arrays:
        return "ball-and-stick"
    if "indices" in arrays:
        return "surface"
    return "points"


def bind(client, kind, arrays):
    """Uploads what a kind needs and returns the bindings for it.

    The mapping from an array's name to the input it feeds is *this script's*,
    not the server's. Nothing infers a role from a name any more: these samples
    happen to call their coordinates "positions", and a sample that called them
    "xyz" would bind exactly the same way with one line changed here.
    """
    # Input id -> the name this script's data happens to use for it. `surface`
    # takes one input, so its arrays are named here for the *geometry filter*
    # that assembles them rather than for the actor itself.
    roles = {
        "points": {"positions": "positions"},
        "surface": {
            "positions": "positions",
            "indices": "indices",
            "normals": "normals",
        },
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
        # Through a filter, not a setting. Colour reaches a consumer as linear
        # RGB, so what turns a scalar field into colours is a `colormap` of its
        # own — which is what lets the same field be shown through a different
        # ramp, or a different field entirely, without touching the actor.
        colours = client.add_filter(
            "colormap", params={"values": iris3d.Bind(held[scalar])}
        )
        params["colour"] = iris3d.Bind(colours["colour"])

    if kind == "surface":
        # A `surface` actor takes one input: geometry somebody assembled. Loose
        # arrays go through the `geometry` filter first, which is deliberately
        # the same path a computed ribbon takes rather than a second one for
        # uploads. The colours are part of that assembly, because two actors
        # over one mesh share the vertex buffer and cannot each paint it.
        shape = client.add_filter("geometry", params=params)
        return {"geometry": iris3d.Bind(shape["geometry"])}
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
        # A place in the tree per sample. The data goes in separately and the
        # actor binds it, so an object is only ever a name and a transform.
        for name in everything:
            handle = client.create_object(name)
            client.set_parent(handle, root)
            client.set_transform(handle, translation=placements[name])

        hydrogen(client, root, cursor)

        # How each sample should look is *this script's* decision, and it is the
        # only thing that can make it — the server no longer offers a mapping
        # from a dataset shape to a kind, because an actor's data is bound to it
        # rather than taken from the object it hangs under. `actor_kinds()` still
        # says what exists, which is what stops this asking for something the
        # build cannot draw.
        available = client.actor_kinds()
        for summary in client.list_objects():
            # The grid already has the volume `hydrogen` asked for, and the root
            # is a grouping node with nothing to draw.
            original = everything.get(summary.name)
            if original is None or summary.actors:
                continue
            wanted = kind_for(original)
            if wanted not in available:
                print(f"skipping {summary.name}: this build cannot draw {wanted}")
                continue
            client.add_actor(
                wanted,
                parent=summary.handle,
                params=bind(client, wanted, everything[summary.name]),
            )

        print(f"\n{'handle':<8}{'object':<20}{'drawn as':<16}")
        print("-" * 46)
        for summary in sorted(client.list_objects(), key=lambda s: s.handle):
            indent = "  " if summary.parent is not None else ""
            print(
                f"{summary.handle:<8}{indent + summary.name:<20}"
                f"{', '.join(r.kind for r in summary.actors) or '-':<16}"
            )

        # Data is its own listing now, because it belongs to no object. Arrays
        # and the meshes filters assembled come back together — one handle
        # space — so each row says which it is.
        held = client.list_data()
        print(f"\n{'handle':<8}{'name':<20}{'type':<28}size")
        print("-" * 68)
        for entry in held:
            if isinstance(entry, iris3d.GeometrySummary):
                # No size: the vertices live on the GPU and are never fetched.
                described = f"mesh[{entry.vertices} v, {entry.triangles} t]"
                size = "-"
            else:
                described = f"{entry.dtype}{list(entry.shape)}"
                size = f"{entry.byte_length / 1024:.1f} KiB"
            print(f"d{entry.handle:<7}{entry.name:<20}{described:<28}{size}")
        arrays = [e for e in held if isinstance(e, iris3d.DataSummary)]
        total = sum(a.byte_length for a in arrays)
        meshes = len(held) - len(arrays)
        print(
            f"\n{total / 1024:.1f} KiB resident across {len(arrays)} arrays, "
            f"plus {meshes} assembled mesh{'' if meshes == 1 else 'es'}"
        )


if __name__ == "__main__":
    main()
