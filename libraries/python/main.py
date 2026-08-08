"""Scratch client: loads the sample datasets into a running iris3d."""

import iris3d
from iris3d import testdata

# Laid out along x so the samples do not overlap. The two tori are the same
# shape by construction, so without this the point cloud sits inside the mesh
# and neither is legible.
LAYOUT = {
    "torus (points)": (-10.0, 0.0, 0.0),
    "torus (mesh)": (0.0, 0.0, 0.0),
    "cantilever beam": (7.0, 0.0, 0.0),
    "benzene": (17.0, 0.0, 0.0),
}


def main():
    # Waits for the app to come up, so this can be launched alongside it.
    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        print("connected")

        # Everything hangs off one empty object, so the whole sample set can be
        # moved — or removed — as a unit.
        root = client.create_object("examples")

        for name, arrays in testdata.examples().items():
            handle = client.upload_object(name, arrays)
            client.set_parent(handle, root)
            client.set_transform(handle, translation=LAYOUT.get(name, (0.0, 0.0, 0.0)))

        print(f"\n{'handle':<8}{'object':<18}{'kind':<11}{'drawn as':<16}arrays")
        print("-" * 78)
        for summary in sorted(client.list_objects(), key=lambda s: s.handle):
            indent = "  " if summary.parent is not None else ""
            arrays = ", ".join(f"{b.name}{list(b.shape)}" for b in summary.buffers)
            print(
                f"{summary.handle:<8}{indent + summary.name:<18}"
                f"{summary.dataset_kind:<11}"
                f"{', '.join(summary.representations) or '-':<16}{arrays}"
            )

        total = sum(s.total_bytes for s in client.list_objects())
        print(f"\n{total / 1024:.1f} KiB resident")


if __name__ == "__main__":
    main()
