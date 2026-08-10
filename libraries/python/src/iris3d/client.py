"""Python client for the iris3d scene service.

The wire contract describes data the way numpy does — a raw little-endian byte
buffer plus a dtype and a shape — so this wrapper is mostly a translation layer
between ``numpy.ndarray`` and ``BufferSpec``/``Chunk``. Other language wrappers
should end up looking much the same.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import grpc
import numpy as np

from .v1.scene_pb2 import (
    ActorHandle,
    ActorInfo,
    ActorKindInfo,
    AddActorRequest,
    BufferSpec,
    Chunk,
    Color,
    ColorSpec,
    CreateObjectRequest,
    DataHandle,
    DataHeader,
    DataInfo,
    DeleteObjectRequest,
    Dimensions,
    Dtype,
    Grid as ProtoGrid,
    ListActorKindsRequest,
    ListActorsRequest,
    ListDataRequest,
    ListObjectsRequest,
    ObjectHandle,
    ObjectHeader,
    ObjectInfo,
    ParamValue,
    Quaternion,
    Range,
    ReleaseDataRequest,
    RemoveActorRequest,
    SetActorRequest,
    SetParentRequest,
    Subset,
    SetTransformRequest,
    UploadDataRequest,
    UploadObjectRequest,
    Vector3,
    VectorValue,
)
from .v1.scene_pb2_grpc import SceneServiceStub

DEFAULT_ADDRESS = "[::1]:50051"

#: Chunk payload size. The server accepts messages up to 8 MiB; staying well
#: under that leaves room for framing overhead and keeps memory churn low.
DEFAULT_CHUNK_BYTES = 1 << 20

#: Default ceiling for :meth:`Client.wait_until_ready`. Generous enough to cover
#: a cold debug-build start of the Bevy app.
DEFAULT_CONNECT_TIMEOUT = 60.0

_TO_PROTO: dict[np.dtype, "Dtype.ValueType"] = {
    np.dtype(np.uint8): Dtype.DTYPE_UINT8,
    np.dtype(np.int8): Dtype.DTYPE_INT8,
    np.dtype(np.uint16): Dtype.DTYPE_UINT16,
    np.dtype(np.int16): Dtype.DTYPE_INT16,
    np.dtype(np.uint32): Dtype.DTYPE_UINT32,
    np.dtype(np.int32): Dtype.DTYPE_INT32,
    np.dtype(np.uint64): Dtype.DTYPE_UINT64,
    np.dtype(np.int64): Dtype.DTYPE_INT64,
    np.dtype(np.float32): Dtype.DTYPE_FLOAT32,
    np.dtype(np.float64): Dtype.DTYPE_FLOAT64,
}

_FROM_PROTO = {proto: dtype for dtype, proto in _TO_PROTO.items()}


def to_proto_dtype(dtype: np.dtype) -> "Dtype.ValueType":
    """Maps a numpy dtype onto its wire equivalent."""
    try:
        return _TO_PROTO[np.dtype(dtype).newbyteorder("=")]
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}") from None


def from_proto_dtype(dtype: "Dtype.ValueType") -> np.dtype:
    """Maps a wire dtype back onto numpy."""
    try:
        return _FROM_PROTO[dtype]
    except KeyError:
        raise ValueError(f"unknown wire dtype {dtype!r}") from None


def _vector3(values: tuple[float, float, float]) -> Vector3:
    """Generated protobuf messages take keyword arguments only."""
    return Vector3(x=values[0], y=values[1], z=values[2])


def _wire_ready(array: np.ndarray) -> np.ndarray:
    """Returns the array as densely packed, little-endian, C-order data."""
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array


@dataclass(frozen=True)
class BufferInfo:
    """Metadata for one array belonging to a scene object."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    byte_length: int


@dataclass(frozen=True)
class ObjectSummary:
    """A scene object as reported by the server, without its contents."""

    handle: int
    name: str
    buffers: tuple[BufferInfo, ...]
    total_bytes: int
    #: Structure inferred by the server: "points", "mesh", "grid", "molecule"
    #: or "raw".
    dataset_kind: str
    #: Everything currently drawing this object's data; empty if nothing does.
    actors: tuple["ActorSummary", ...]
    #: Parent handle in the scene tree, or None for a root object.
    parent: int | None


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned grid.

    Sample ``(i, j, k)`` sits at ``origin + (i, j, k) * spacing``, with ``i``
    varying fastest — the same C order numpy uses, so a ``(nx, ny, nz)`` array
    ravels straight onto it.

    Pass one to :meth:`Client.upload_object` and the buffers become fields over
    the grid. Send no positions: a 256³ volume states its geometry in these nine
    numbers instead of 50 million coordinates, which is the point of the type.
    """

    dims: tuple[int, int, int]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        """Rejects a grid that cannot describe anything.

        Checked here rather than at upload because ``upload_messages`` is a
        generator: an exception raised inside it surfaces from grpc as
        ``UNKNOWN: Exception iterating requests!``, which says nothing about
        what was wrong. Failing at construction points at the line that built
        the grid.
        """
        if len(self.dims) != 3 or any(n < 1 for n in self.dims):
            raise ValueError(
                f"grid dims must be three counts of at least one, got {self.dims}"
            )
        if len(self.spacing) != 3 or any(s <= 0.0 for s in self.spacing):
            raise ValueError(
                "grid spacing must be three values greater than zero, "
                f"got {self.spacing}"
            )
        if len(self.origin) != 3:
            raise ValueError(f"grid origin must be three values, got {self.origin}")

    @property
    def point_count(self) -> int:
        """Samples in the grid."""
        return self.dims[0] * self.dims[1] * self.dims[2]

    @property
    def cell_count(self) -> int:
        """Cells between the samples. An axis of one sample spans none."""
        x, y, z = (max(0, n - 1) for n in self.dims)
        return x * y * z

    def to_proto(self) -> ProtoGrid:
        return ProtoGrid(
            origin=_vector3(self.origin),
            spacing=_vector3(self.spacing),
            dims=Dimensions(x=self.dims[0], y=self.dims[1], z=self.dims[2]),
        )


@dataclass(frozen=True)
class Coloring:
    """How an actor takes its colour."""

    #: "viridis", "cool-warm", "grayscale" or "element".
    map: str = "viridis"
    #: Value range the map spans. None autoscales to the bound array's own range.
    range: tuple[float, float] | None = None
    #: sRGB, used when no colour array is bound.
    flat: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SubsetSummary:
    """An actor's selection, described without returning its values."""

    #: "indices" or "mask".
    encoding: str
    #: "point" or "cell".
    association: str
    #: How many elements the selection keeps.
    selected: int


@dataclass(frozen=True)
class ActorSummary:
    """One way something is being drawn."""

    handle: int
    #: Registered kind id, e.g. "points" or "ball-and-stick".
    kind: str
    #: Handle of the object whose data is drawn.
    source: int
    #: Handle of the object whose transform is inherited — usually ``source``.
    parent: int | None
    #: Complete and in range, whatever was sent to produce it.
    params: dict[str, float | bool]
    coloring: Coloring
    visible: bool
    #: How much of the source is drawn, or None for all of it.
    subset: SubsetSummary | None = None


@dataclass(frozen=True)
class ParamInfo:
    """One setting or input an actor kind accepts."""

    id: str
    label: str
    #: "float", "bool", "choice", "array" or "vector".
    type: str
    #: Absent for an array input: there is no default array, so it starts unbound.
    default: float | bool | str | None
    #: Allowed range, for float parameters only.
    range: tuple[float, float] | None = None
    logarithmic: bool = False
    #: The permitted values, for choice parameters only.
    options: tuple[str, ...] = ()
    #: Element types this input accepts, for arrays only. Empty accepts any.
    dtypes: tuple[np.dtype, ...] = ()
    #: Declared shape, for arrays only. 0 accepts any length on that axis, so
    #: positions read ``(0, 3)`` and a scalar field ``(0,)``.
    shape: tuple[int, ...] = ()
    #: Whether the kind can draw without it, for arrays only.
    required: bool = False
    #: How many numbers it takes, for vectors only.
    components: int = 0
    #: Whole numbers only, for vectors only. True for counts such as grid dims.
    integral: bool = False


@dataclass(frozen=True)
class Bind:
    """Binds an uploaded array to one of an actor kind's inputs.

    Wrapped rather than passed as a bare handle so it cannot be mistaken for a
    slider value::

        data = client.upload_data({"xyz": positions})
        client.add_actor(obj, "points", params={"positions": iris3d.Bind(data["xyz"])})
    """

    handle: int


@dataclass(frozen=True)
class DataSummary:
    """One array the scene holds, described without its contents."""

    handle: int
    #: The label it was uploaded under. Not a role — see :meth:`Client.upload_data`.
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    byte_length: int


@dataclass(frozen=True)
class ActorKindSummary:
    """A way of drawing that the running server supports."""

    id: str
    label: str
    params: tuple[ParamInfo, ...]


def _param_value(value: float | bool | str) -> ParamValue:
    """Wraps a Python value for the wire.

    ``bool`` is checked first: it is a subclass of ``int``, so testing for a
    number would swallow it and send True as 1.0 — which the server would then
    reject as the wrong type for a boolean parameter.
    """
    if isinstance(value, Bind):
        return ParamValue(data=DataHandle(id=value.handle))
    if isinstance(value, bool):
        return ParamValue(flag=value)
    if isinstance(value, (int, float)):
        return ParamValue(number=float(value))
    if isinstance(value, str):
        return ParamValue(text=value)
    # Before the generic sequence check would catch a string, which is why that
    # is tested above: "xyz" is a sequence of three characters.
    if isinstance(value, (tuple, list, np.ndarray)):
        components = [float(component) for component in value]
        return ParamValue(vector=VectorValue(components=components))
    raise TypeError(
        "parameter values must be a number, a bool, a string, a sequence of "
        f"numbers or a Bind, not {type(value).__name__}"
    )


def _read_param(value: ParamValue) -> float | bool | str:
    """Unwraps a value from the wire, keeping its type."""
    match value.WhichOneof("value"):
        case "flag":
            return value.flag
        case "text":
            return value.text
        case "data":
            # Comes back wrapped, so a round trip keeps an array distinguishable
            # from a number that happens to equal its handle.
            return Bind(value.data.id)
        case "vector":
            return tuple(value.vector.components)
        case _:
            return value.number


def _params(params: Mapping[str, float | bool | str] | None) -> dict[str, ParamValue]:
    return {key: _param_value(value) for key, value in (params or {}).items()}


def _color_spec(coloring: Coloring) -> ColorSpec:
    spec = ColorSpec(map=coloring.map)
    if coloring.range is not None:
        spec.range.CopyFrom(Range(low=coloring.range[0], high=coloring.range[1]))
    if coloring.flat is not None:
        spec.flat.CopyFrom(Color(r=coloring.flat[0], g=coloring.flat[1], b=coloring.flat[2]))
    return spec


def _coloring(spec: ColorSpec) -> Coloring:
    return Coloring(
        map=spec.map,
        range=(spec.range.low, spec.range.high) if spec.HasField("range") else None,
        flat=(spec.flat.r, spec.flat.g, spec.flat.b) if spec.HasField("flat") else None,
    )


_ENCODINGS = {
    Subset.ENCODING_INDICES: "indices",
    Subset.ENCODING_MASK: "mask",
}
_ASSOCIATIONS = {
    Subset.ASSOCIATION_PER_POINT: "point",
    Subset.ASSOCIATION_PER_CELL: "cell",
}


def _subset(selection: np.ndarray, *, per_cell: bool = False) -> Subset:
    """Packs a numpy selection for the wire.

    The encoding follows the dtype, because that is what the caller already
    expressed: a boolean array is a mask over every element, an integer array
    names the elements to keep. Asking for both would be one more way to
    disagree with yourself.
    """
    selection = np.ascontiguousarray(selection)
    if selection.ndim != 1:
        raise ValueError(f"a subset must be one-dimensional, got shape {selection.shape}")

    if selection.dtype == np.bool_:
        encoding = Subset.ENCODING_MASK
        selection = selection.view(np.uint8)
    elif np.issubdtype(selection.dtype, np.integer):
        encoding = Subset.ENCODING_INDICES
    else:
        raise TypeError(
            "a subset must be a boolean mask or an integer index array, "
            f"not {selection.dtype}"
        )

    return Subset(
        data=_wire_ready(selection).tobytes(),
        dtype=to_proto_dtype(selection.dtype),
        encoding=encoding,
        association=(
            Subset.ASSOCIATION_PER_CELL if per_cell else Subset.ASSOCIATION_PER_POINT
        ),
    )


def _actor(info: ActorInfo) -> ActorSummary:
    return ActorSummary(
        handle=info.handle.id,
        kind=info.kind,
        source=info.source.id,
        parent=info.parent.id if info.HasField("parent") else None,
        params={key: _read_param(value) for key, value in info.params.items()},
        coloring=_coloring(info.color),
        visible=info.visible,
        subset=(
            SubsetSummary(
                encoding=_ENCODINGS.get(info.subset.encoding, "unknown"),
                association=_ASSOCIATIONS.get(info.subset.association, "point"),
                selected=info.subset.selected,
            )
            if info.HasField("subset")
            else None
        ),
    )


def _data(info: DataInfo) -> DataSummary:
    return DataSummary(
        handle=info.handle.id,
        name=info.spec.name,
        dtype=from_proto_dtype(info.spec.dtype),
        shape=tuple(info.spec.shape),
        byte_length=info.spec.byte_length,
    )


def _kind(info: ActorKindInfo) -> ActorKindSummary:
    params = []
    for spec in info.params:
        kind = spec.WhichOneof("kind")
        if kind == "flag":
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="bool",
                    default=spec.flag.default_value,
                )
            )
        elif kind == "vector":
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="vector",
                    default=tuple(spec.vector.default_value),
                    range=(spec.vector.min, spec.vector.max),
                    components=spec.vector.components,
                    integral=spec.vector.integral,
                )
            )
        elif kind == "choice":
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="choice",
                    default=spec.choice.default_value,
                    options=tuple(spec.choice.options),
                )
            )
        elif kind == "array":
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="array",
                    # No default array exists, so an input starts unbound.
                    default=None,
                    dtypes=tuple(from_proto_dtype(d) for d in spec.array.dtypes),
                    shape=tuple(spec.array.shape),
                    required=spec.array.required,
                )
            )
        else:
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="float",
                    default=spec.number.default_value,
                    range=(spec.number.min, spec.number.max),
                    logarithmic=spec.number.logarithmic,
                )
            )
    return ActorKindSummary(
        id=info.id,
        label=info.label,
        params=tuple(params),
    )


def _summary(info: ObjectInfo) -> ObjectSummary:
    return ObjectSummary(
        handle=info.handle.id,
        name=info.name,
        buffers=tuple(
            BufferInfo(
                name=spec.name,
                dtype=from_proto_dtype(spec.dtype),
                shape=tuple(spec.shape),
                byte_length=spec.byte_length,
            )
            for spec in info.buffers
        ),
        total_bytes=info.total_bytes,
        dataset_kind=info.dataset_kind,
        actors=tuple(_actor(actor) for actor in info.actors),
        parent=info.parent.id if info.HasField("parent") else None,
    )


def _declare(
    arrays: Mapping[str, np.ndarray], chunk_bytes: int
) -> tuple[list[np.ndarray], list[BufferSpec]]:
    """Validates arrays and describes them for a header.

    Shared by both upload streams, so an object upload and a bare data upload
    cannot disagree about what a well-formed declaration is.
    """
    if not arrays:
        raise ValueError("an upload needs at least one array")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    prepared: list[np.ndarray] = []
    specs: list[BufferSpec] = []
    for buffer_name, array in arrays.items():
        array = _wire_ready(np.asarray(array))
        if array.size == 0:
            raise ValueError(f"array {buffer_name!r} is empty")
        prepared.append(array)
        specs.append(
            BufferSpec(
                name=buffer_name,
                dtype=to_proto_dtype(array.dtype),
                shape=list(array.shape),
                byte_length=array.nbytes,
            )
        )
    return prepared, specs


def _payload_chunks(
    prepared: list[np.ndarray], chunk_bytes: int
) -> Iterator[Chunk]:
    """Slices every array into wire-sized chunks, in declaration order."""
    for index, array in enumerate(prepared):
        payload = memoryview(array).cast("B")
        for offset in range(0, len(payload), chunk_bytes):
            yield Chunk(
                buffer_index=index,
                offset=offset,
                data=bytes(payload[offset : offset + chunk_bytes]),
            )


def data_messages(
    arrays: Mapping[str, np.ndarray],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[UploadDataRequest]:
    """Builds the request stream for a bare data upload.

    No object, no name, no grid — just arrays. Names here are labels for the
    inventory, not roles: what an array means to a representation is settled
    when it is bound to one, not guessed from what it was called.
    """
    prepared, specs = _declare(arrays, chunk_bytes)
    yield UploadDataRequest(header=DataHeader(arrays=specs))
    for chunk in _payload_chunks(prepared, chunk_bytes):
        yield UploadDataRequest(chunk=chunk)


def upload_messages(
    name: str,
    arrays: Mapping[str, np.ndarray],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    grid: "Grid | None" = None,
) -> Iterator[UploadObjectRequest]:
    """Builds the request stream for one object.

    Yields a header declaring every buffer, then the buffers' bytes in order.
    Exposed separately from :meth:`Client.upload_object` so callers can wrap or
    inspect the stream — for progress reporting, say.

    ``grid`` declares the buffers to be fields sampled over a regular grid. It
    is the one structure the server cannot infer, because a grid's sample
    positions are implicit and no array gives it away.
    """
    prepared, specs = _declare(arrays, chunk_bytes)
    header = ObjectHeader(name=name, buffers=specs)
    if grid is not None:
        header.grid.CopyFrom(grid.to_proto())
    yield UploadObjectRequest(header=header)
    for chunk in _payload_chunks(prepared, chunk_bytes):
        yield UploadObjectRequest(chunk=chunk)


class Client:
    """A connection to a running iris3d instance."""

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        wait_timeout: float | None = None,
    ) -> None:
        """Opens a channel to iris3d.

        The channel is lazy, so nothing connects until the first call. Pass
        ``wait_timeout`` to block until the server is actually reachable —
        useful when the app is still starting, as when an editor launches both
        at once.
        """
        self._address = address
        self._channel = grpc.insecure_channel(address)
        self._scene = SceneServiceStub(self._channel)
        if wait_timeout is not None:
            self.wait_until_ready(wait_timeout)

    def wait_until_ready(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        """Blocks until the server accepts connections.

        Without this, calls made while iris3d is still booting fail
        immediately with ``UNAVAILABLE`` rather than waiting.
        """
        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            raise ConnectionError(
                f"iris3d was not reachable at {self._address} within {timeout:g}s"
            ) from None

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

    def upload_object(
        self,
        name: str,
        arrays: Mapping[str, np.ndarray],
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        *,
        grid: Grid | None = None,
    ) -> int:
        """Uploads one object and returns its handle.

        ``arrays`` maps buffer names to arrays. Conventional names are
        ``positions`` (float32, ``[n, 3]``), ``colors`` (uint8, ``[n, 3]`` or
        ``[n, 4]``), ``indices`` (uint32, ``[m, 3]``) and ``normals``
        (float32, ``[n, 3]``). The server infers the structure from them.

        Pass ``grid`` for a regular grid, which is the one structure the names
        cannot express. Then send no positions and no indices — every buffer is
        a field over the samples::

            nx, ny, nz = 64, 64, 64
            client.upload_object(
                "density",
                {"density": values.ravel()},
                grid=iris3d.Grid(dims=(nx, ny, nz), spacing=(0.1, 0.1, 0.1)),
            )

        A field of one value per sample is per-point. A field of one value per
        cell — ``(nx-1) * (ny-1) * (nz-1)`` of them — is per-cell. This is the
        only upload where the server can tell the two apart.
        """
        response = self._scene.UploadObject(
            upload_messages(name, arrays, chunk_bytes, grid=grid)
        )
        return response.handle.id

    def upload_data(
        self,
        arrays: Mapping[str, np.ndarray],
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> dict[str, int]:
        """Uploads arrays on their own, returning a handle for each by name.

        Nothing is created and nothing is drawn — this puts numbers in the
        scene and stops there. Bind the handles to an actor's inputs to see
        them, and upload an array once to feed as many actors as you like.

        The names are labels, not roles::

            data = client.upload_data({"xyz": positions, "t": temperature})
            # data == {"xyz": 7, "t": 8}

        Use :meth:`release_data` when you are done with them. Nothing else
        will: an array that no actor reads is still held until you say so.
        """
        response = self._scene.UploadData(data_messages(arrays, chunk_bytes))
        return {
            info.spec.name: info.handle.id for info in response.arrays
        }

    def list_data(self) -> list[DataSummary]:
        """Every array currently held, whether or not anything draws it."""
        response = self._scene.ListData(ListDataRequest())
        return [_data(info) for info in response.arrays]

    def release_data(self, *handles: int) -> tuple[int, ...]:
        """Forgets arrays, returning the handles that were actually held.

        The bytes go once nothing refers to them, so releasing an array an
        actor still reads frees nothing until that actor goes too.
        """
        response = self._scene.ReleaseData(
            ReleaseDataRequest(handles=[DataHandle(id=h) for h in handles])
        )
        return tuple(h.id for h in response.released)

    def create_object(self, name: str) -> int:
        """Creates an object holding no data and returns its handle.

        Objects form a tree and anything may parent anything, so an empty
        object is how you make a pure grouping node: parent others to it with
        :meth:`set_parent` and they share its transform.
        """
        response = self._scene.CreateObject(CreateObjectRequest(name=name))
        return response.handle.id

    def set_parent(
        self,
        handle: int,
        parent: int | None,
        *,
        keep_world_transform: bool = False,
    ) -> None:
        """Moves an object within the tree; ``parent=None`` detaches it.

        Raises ``grpc.RpcError`` with ``FAILED_PRECONDITION`` if the move would
        make a cycle, and ``NOT_FOUND`` if either handle is unknown. Pass
        ``keep_world_transform`` to stop the object shifting by the new
        parent's transform.
        """
        self._scene.SetParent(
            SetParentRequest(
                handle=ObjectHandle(id=handle),
                parent=None if parent is None else ObjectHandle(id=parent),
                keep_world_transform=keep_world_transform,
            )
        )

    def set_transform(
        self,
        handle: int,
        *,
        translation: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float, float] | None = None,
        scale: tuple[float, float, float] | float | None = None,
    ) -> None:
        """Sets an object's placement relative to its parent.

        Anything left as ``None`` is untouched, so you can move an object
        without disturbing its rotation. ``rotation`` is a quaternion in
        ``(x, y, z, w)`` order; ``scale`` accepts a single number for a uniform
        scale.
        """
        if isinstance(scale, (int, float)):
            scale = (float(scale),) * 3
        self._scene.SetTransform(
            SetTransformRequest(
                handle=ObjectHandle(id=handle),
                translation=None if translation is None else _vector3(translation),
                rotation=None
                if rotation is None
                else Quaternion(
                    x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3]
                ),
                scale=None if scale is None else _vector3(scale),
            )
        )

    def list_objects(self) -> list[ObjectSummary]:
        response = self._scene.ListObjects(ListObjectsRequest())
        return [_summary(info) for info in response.objects]

    def delete_object(self, handle: int, *, recursive: bool = False) -> tuple[int, ...]:
        """Removes an object, returning the handles actually removed.

        Descendants are detached and become roots unless ``recursive`` is set,
        so deleting an object never destroys data the caller did not name.
        Returns an empty tuple if the handle was already gone.
        """
        response = self._scene.DeleteObject(
            DeleteObjectRequest(handle=ObjectHandle(id=handle), recursive=recursive)
        )
        return tuple(h.id for h in response.removed)

    def add_actor(
        self,
        source: int,
        kind: str,
        *,
        parent: int | None = None,
        params: Mapping[str, float | bool | str] | None = None,
        coloring: Coloring | None = None,
        subset: np.ndarray | None = None,
        per_cell: bool = False,
    ) -> ActorSummary:
        """Draws an object an additional way.

        Adds rather than replaces: an object may be drawn several ways at once,
        each configured on its own.

        ``kind`` is required, and an upload draws nothing until you call this.
        The server has no opinion on how a dataset should look, so the choice
        belongs here. :meth:`actor_kinds` reports what this build supports and
        which datasets each one can draw.

        ``parent`` is the object whose *transform* is inherited, as distinct
        from ``source``, whose *data* is drawn. Passing a different object
        renders one dataset in two places without uploading it twice::

            ghost = client.create_object("ghost")
            client.set_transform(ghost, translation=(10, 0, 0))
            client.add_actor(protein, parent=ghost)

        Parameters left out take the kind's default, not the value some other
        actor happens to have.

        ``subset`` draws only part of the source — a boolean mask over every
        element, or an integer array of the elements to keep. This is what
        makes several actors worth having: one structure shown as cartoon over
        its protein and ball-and-stick over its ligand is two actors with two
        subsets. Selections are computed here rather than described to the
        server, so anything numpy can express works::

            client.add_actor(mesh, subset=positions[:, 2] > 0)

        A mesh cell survives only when all of its corners do, and a bond only
        when both its atoms do, so a cut leaves a clean boundary rather than
        stretched or dangling geometry.
        """
        request = AddActorRequest(
            source=ObjectHandle(id=source),
            kind=kind,
            params=_params(params),
        )
        if parent is not None:
            request.parent.CopyFrom(ObjectHandle(id=parent))
        if coloring is not None:
            request.color.CopyFrom(_color_spec(coloring))
        if subset is not None:
            request.subset.CopyFrom(_subset(subset, per_cell=per_cell))
        return _actor(self._scene.AddActor(request).actor)

    def set_actor(
        self,
        handle: int,
        params: Mapping[str, float | bool | str] | None = None,
        *,
        coloring: Coloring | None = None,
        visible: bool | None = None,
        subset: np.ndarray | None = None,
        per_cell: bool = False,
        clear_subset: bool = False,
    ) -> ActorSummary:
        """Changes an actor, leaving anything unnamed alone.

        Parameters are merged, so passing one setting keeps the rest — the
        opposite of :meth:`add_actor`, where an absent parameter takes its
        default. ``coloring`` is all-or-nothing: passing it replaces the
        colouring outright.

        Out-of-range values are clamped rather than rejected, so a slider driven
        past its limit does not raise.

        Omitting ``subset`` leaves the selection alone; ``clear_subset=True``
        goes back to drawing the whole dataset. The two are separate because
        "unchanged" and "cleared" both have to be expressible.
        """
        if subset is not None and clear_subset:
            raise ValueError("pass a subset or clear_subset, not both")

        request = SetActorRequest(
            handle=ActorHandle(id=handle),
            params=_params(params),
            clear_subset=clear_subset,
        )
        if coloring is not None:
            request.color.CopyFrom(_color_spec(coloring))
        if visible is not None:
            request.visible = visible
        if subset is not None:
            request.subset.CopyFrom(_subset(subset, per_cell=per_cell))
        return _actor(self._scene.SetActor(request).actor)

    def remove_actor(self, handle: int) -> bool:
        """Stops drawing something one way, leaving the object and its data.

        Returns False if the handle was already gone.
        """
        response = self._scene.RemoveActor(RemoveActorRequest(handle=ActorHandle(id=handle)))
        return response.removed

    def list_actors(self, source: int | None = None) -> list[ActorSummary]:
        """Lists actors, optionally only those drawing one object."""
        request = ListActorsRequest()
        if source is not None:
            request.source.CopyFrom(ObjectHandle(id=source))
        response = self._scene.ListActors(request)
        return [_actor(info) for info in response.actors]

    def actor_kinds(self) -> dict[str, ActorKindSummary]:
        """The ways of drawing this server supports, keyed by kind id.

        Kinds come from whichever rendering backends the server was built with,
        so ask rather than assuming: a hardcoded list here would eventually
        offer something that silently does nothing.

        A kind's ``params`` include its array inputs, each saying what element
        types and shape it takes and whether the kind can draw without it. That
        is what to bind an uploaded array to::

            wanted = client.actor_kinds()["points"]
            [p.id for p in wanted.params if p.type == "array" and p.required]
        """
        response = self._scene.ListActorKinds(ListActorKindsRequest())
        return {key: _kind(info) for key, info in response.kinds.items()}
