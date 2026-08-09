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
    return placements


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


def main():
    # Waits for the app to come up, so this can be launched alongside it.
    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        print("connected")

        # Everything hangs off one empty object, so the whole sample set can be
        # moved — or removed — as a unit.
        root = client.create_object("examples")

        everything = datasets()
        placements = layout(everything)
        for name, arrays in everything.items():
            handle = client.upload_object(name, arrays)
            client.set_parent(handle, root)
            client.set_transform(handle, translation=placements[name])

        print(f"\n{'handle':<8}{'object':<18}{'kind':<11}{'drawn as':<16}arrays")
        print("-" * 78)
        for summary in sorted(client.list_objects(), key=lambda s: s.handle):
            indent = "  " if summary.parent is not None else ""
            arrays = ", ".join(f"{b.name}{list(b.shape)}" for b in summary.buffers)
            print(
                f"{summary.handle:<8}{indent + summary.name:<18}"
                f"{summary.dataset_kind:<11}"
                f"{', '.join(r.kind for r in summary.representations) or '-':<16}{arrays}"
            )

        total = sum(s.total_bytes for s in client.list_objects())
        print(f"\n{total / 1024:.1f} KiB resident")


if __name__ == "__main__":
    main()
