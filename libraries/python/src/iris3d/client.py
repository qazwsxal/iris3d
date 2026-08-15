"""Python client for the iris3d scene service.

The wire contract describes data the way numpy does — a raw little-endian byte
buffer plus a dtype and a shape — so this wrapper is mostly a translation layer
between ``numpy.ndarray`` and ``BufferSpec``/``Chunk``. Other language wrappers
should end up looking much the same.

Text is the one array that is not bytes. A numpy array of strings — or a plain
list of them — uploads as a string array, carried whole in the header rather
than chunked. It is how labelling data travels: which chain each atom belongs
to is an integer array, and what those chains are *called* is a string array
beside it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

import grpc
import numpy as np

from .v1.scene_pb2 import (
    ActorHandle,
    ActorInfo,
    ActorKindInfo,
    AddActorRequest,
    AddFilterRequest,
    BufferSpec,
    Chunk,
    CreateObjectRequest,
    DataHandle,
    DataHeader,
    DataInfo,
    DeleteObjectRequest,
    Dtype,
    FilterHandle,
    FilterInfo,
    FilterKindInfo,
    ListActorKindsRequest,
    ListActorsRequest,
    ListDataRequest,
    ListFilterKindsRequest,
    ListFiltersRequest,
    ListObjectsRequest,
    ObjectHandle,
    ObjectHandles,
    ObjectInfo,
    ParamValue,
    Quaternion,
    ReleaseDataRequest,
    RemoveActorRequest,
    RemoveFilterRequest,
    SetActorRequest,
    SetFilterRequest,
    SetParentRequest,
    SetTransformRequest,
    Subset,
    Unset as ProtoUnset,
    UploadDataRequest,
    Vector3,
    VectorValue,
)
from .v1.scene_pb2_grpc import SceneServiceStub


def _maybe(handle: int | None) -> tuple[int, ...]:
    """One handle as a sequence, or none at all."""
    return () if handle is None else (handle,)

DEFAULT_ADDRESS = "[::1]:50051"

#: Chunk payload size. The server accepts messages up to 8 MiB; staying well
#: under that leaves room for framing overhead and keeps memory churn low.
DEFAULT_CHUNK_BYTES = 1 << 20

#: Default ceiling for :meth:`Client.wait_until_ready`. Generous enough to cover
#: a cold debug-build start of the Bevy app.
DEFAULT_CONNECT_TIMEOUT = 60.0

_TO_PROTO: dict[np.dtype, Dtype] = {
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
    # Text, whose numpy side is `object` rather than a width like `<U5`. The
    # wire carries each string at its own length, so no fixed width is the
    # truthful answer coming back: a `<U5` array uploads fine, but describing
    # what the server holds as `<U5` would claim a padding the wire dropped.
    np.dtype(object): Dtype.DTYPE_STRING,
}

_FROM_PROTO = {proto: dtype for dtype, proto in _TO_PROTO.items()}


def to_proto_dtype(dtype: np.dtype) -> Dtype:
    """Maps a numpy dtype onto its wire equivalent."""
    try:
        return _TO_PROTO[np.dtype(dtype).newbyteorder("=")]
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}") from None


def from_proto_dtype(dtype: Dtype) -> np.dtype:
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
class ObjectSummary:
    """A place in the scene tree, as reported by the server.

    No buffers and no byte count: an object holds no data. Ask
    :meth:`Client.list_data` what is resident.
    """

    handle: int
    name: str
    #: Everything drawing here; empty if nothing does.
    actors: tuple[ActorSummary, ...]
    #: Parent handle in the scene tree, or None for a root object.
    parent: int | None


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned grid.

    Sample ``(i, j, k)`` sits at ``origin + (i, j, k) * spacing``, with ``i``
    varying fastest — the same C order numpy uses, so a ``(nx, ny, nz)`` array
    ravels straight onto it.

    Bind ``dims``, ``origin`` and ``spacing`` to a volume actor. A 256³ volume
    states its geometry in these nine numbers rather than 50 million
    coordinates, which is the point of the type.
    """

    dims: tuple[int, int, int]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        """Rejects a grid that cannot describe anything.

        Failing at construction points at the line that built the grid, rather
        than at the call that binds it.
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
    #: Handles of every object it is drawn under, in order. Several means one
    #: drawing appearing in several places — one mesh, one set of settings, and
    #: every copy changes together. Empty draws nothing, which is where
    #: deleting the last object it was under leaves it; the actor is untouched
    #: and ``set_actor(handle, parents=[...])`` puts it back on screen.
    parents: tuple[int, ...]
    #: Complete and in range, whatever was sent to produce it.
    params: dict[str, RawParamValue]
    #: The setting, not whether anything reaches the screen. A hidden object
    #: above it does not show here, and neither does being detached — a
    #: detached actor is not drawn whatever this says.
    visible: bool
    #: How much of the bound data is drawn, or None for all of it.
    subset: SubsetSummary | None = None


@dataclass(frozen=True)
class ParamInfo:
    """One setting or input an actor kind accepts."""

    id: str
    label: str
    #: "float", "bool", "choice", "array", "geometry" or "vector".
    type: str
    #: Absent for an input: there is no default array or mesh, so it starts
    #: unbound.
    default: float | bool | str | tuple[float, ...] | tuple[int, ...] | None
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
    #: Whether the kind can draw without it, for both sorts of input.
    required: bool = False
    #: How many numbers it takes, for vectors only.
    components: int = 0
    #: Whole numbers only, for vectors only. True for counts such as grid dims.
    integral: bool = False


@dataclass(frozen=True)
class Bind:
    """Binds a handle to one of a kind's inputs.

    Wrapped rather than passed as a bare handle so it cannot be mistaken for a
    slider value::

        data = client.upload_data({"xyz": positions})
        client.add_actor(obj, "points", params={"positions": iris3d.Bind(data["xyz"])})

    The same wrapper binds geometry, because a mesh is named by a handle from
    the same sequence an array is.

    Its opposite is :data:`Unset`, which lets an optional input go again.
    """

    handle: int


@dataclass(frozen=True)
class _Unset:
    """The type of :data:`Unset`."""


#: Takes the binding off an optional input, leaving it as though nothing had
#: ever been bound::
#:
#:     client.set_actor(actor, params={"colour": iris3d.Unset})
#:
#: Needed as a value rather than as an omission because ``set_actor`` and
#: ``set_filter`` take a *partial* map: leaving a parameter out means "leave it
#: alone", so absence cannot also mean "clear it".
#:
#: A required input is refused, with the same error as never having bound it.
Unset = _Unset()


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
class GeometrySummary:
    """One mesh the scene holds, described without its vertices.

    Produced by a filter — the ``geometry`` kind assembles arrays into one — and
    bound to an actor's geometry input. Every actor bound to it references the
    same vertex buffers rather than building its own, which is why two ways of
    drawing one ribbon cost one upload.

    The vertices stay on the GPU. There is no fetch for them.
    """

    handle: int
    #: The output it came from, e.g. "geometry".
    name: str
    vertices: int
    triangles: int
    #: A normal per vertex. Read by lighting and by a glass shell.
    normals: bool
    #: A linear RGB colour per vertex, already mapped.
    colours: bool


@dataclass(frozen=True)
class ActorKindSummary:
    """A way of drawing that the running server supports."""

    id: str
    label: str
    params: tuple[ParamInfo, ...]


@dataclass(frozen=True)
class OutputInfo:
    """One thing a filter kind writes."""

    id: str
    label: str
    #: "array" or "geometry".
    type: str
    #: Element type, for array outputs only.
    dtype: np.dtype | None = None
    #: Declared shape, for array outputs only, where 0 is an axis decided when
    #: it runs. A colour output reads ``(0, 3)``: three components per element,
    #: however many elements there turn out to be.
    #:
    #: A geometry output declares nothing. Which attributes a run produces
    #: depends on what was bound to it, so the answer comes back in
    #: :class:`GeometrySummary` afterwards rather than being promised here.
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class FilterKindSummary:
    """A way of deriving data that the server supports."""

    id: str
    label: str
    #: Settings and array inputs together, as for an actor kind.
    params: tuple[ParamInfo, ...]
    outputs: tuple[OutputInfo, ...]


@dataclass(frozen=True)
class FilterSummary:
    """One filter, and where to find what it makes."""

    handle: int
    #: Registered kind id, e.g. "colormap".
    kind: str
    params: dict[str, float | bool | str | int | tuple[float, ...]]
    #: Data handle per output id, in the kind's declaration order.
    #:
    #: **These are what you bind.** They belong to the filter and live as long
    #: as it does: re-running rewrites the arrays behind them rather than
    #: replacing them, so a binding made once stays valid.
    outputs: dict[str, int]

    def __getitem__(self, output: str) -> int:
        """The handle for one output, so binding reads as one expression::

        colours = client.add_filter("colormap", params={"values": Bind(field)})
        client.add_actor("surface", params={"colour": Bind(colours["colour"])})
        """
        return self.outputs[output]


def _param_value(value: RawParamValue) -> ParamValue:
    """Wraps a Python value for the wire.

    ``bool`` is checked first: it is a subclass of ``int``, so testing for a
    number would swallow it and send True as 1.0 — which the server would then
    reject as the wrong type for a boolean parameter.
    """
    if isinstance(value, Bind):
        return ParamValue(data=DataHandle(id=value.handle))
    if isinstance(value, _Unset):
        return ParamValue(unset=ProtoUnset())
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

RawParamValue = float | bool | str | tuple[float, ...] | Bind | _Unset

def _read_param(value: ParamValue) -> RawParamValue:
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
        case "unset":
            # Only ever travels inbound, as an instruction. A map coming back
            # says what things are set to, and cleared is spelled there by the
            # key being absent — so this is the server contradicting itself
            # rather than a value to hand on as a number.
            raise ValueError("the server reported a parameter as unset")
        case _:
            return value.number


def _params(params: Mapping[str, RawParamValue] | None) -> dict[str, ParamValue]:
    return {key: _param_value(value) for key, value in (params or {}).items()}


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
        parents=tuple(handle.id for handle in info.parents),
        params={key: _read_param(value) for key, value in info.params.items()},
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


def _data(info: DataInfo) -> DataSummary | GeometrySummary:
    """Reads one entry of a listing, which is an array or a mesh.

    One handle space, so the two arrive in one listing and are told apart by
    which arm of the oneof is set.
    """
    if info.WhichOneof("spec") == "geometry":
        return GeometrySummary(
            handle=info.handle.id,
            name=info.geometry.name,
            vertices=info.geometry.vertices,
            triangles=info.geometry.triangles,
            normals=info.geometry.normals,
            colours=info.geometry.colours,
        )
    return DataSummary(
        handle=info.handle.id,
        name=info.buffer.name,
        dtype=from_proto_dtype(info.buffer.dtype),
        shape=tuple(info.buffer.shape),
        byte_length=info.buffer.byte_length,
    )


def _param_infos(specs) -> tuple[ParamInfo, ...]:
    """Reads a kind's declared parameters.

    Shared by actor kinds and filter kinds: both declare their settings and
    their array inputs with the same ``ParamSpec``, so a caller that can read
    one listing can read the other.
    """
    params = []
    for spec in specs:
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
        elif kind == "geometry":
            params.append(
                ParamInfo(
                    id=spec.id,
                    label=spec.label,
                    type="geometry",
                    # As for an array: nothing to default to, so it starts
                    # unbound. Nothing to declare either — the kind reads what
                    # it was given rather than demanding attributes in advance.
                    default=None,
                    required=spec.geometry.required,
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
    return tuple(params)


def _kind(info: ActorKindInfo) -> ActorKindSummary:
    return ActorKindSummary(
        id=info.id,
        label=info.label,
        params=_param_infos(info.params),
    )


def _output(output) -> OutputInfo:
    if output.WhichOneof("kind") == "geometry":
        return OutputInfo(id=output.id, label=output.label, type="geometry")
    return OutputInfo(
        id=output.id,
        label=output.label,
        type="array",
        dtype=from_proto_dtype(output.array.dtype),
        shape=tuple(output.array.shape),
    )


def _filter_kind(info: FilterKindInfo) -> FilterKindSummary:
    return FilterKindSummary(
        id=info.id,
        label=info.label,
        params=_param_infos(info.params),
        outputs=tuple(_output(output) for output in info.outputs),
    )


def _filter(info: FilterInfo) -> FilterSummary:
    return FilterSummary(
        handle=info.handle.id,
        kind=info.kind,
        params={key: _read_param(value) for key, value in info.params.items()},
        outputs={output.id: output.handle.id for output in info.outputs},
    )


def _summary(info: ObjectInfo) -> ObjectSummary:
    return ObjectSummary(
        handle=info.handle.id,
        name=info.name,
        actors=tuple(_actor(actor) for actor in info.actors),
        parent=info.parent.id if info.HasField("parent") else None,
    )


#: numpy kinds that mean text: fixed-width unicode, byte strings, and object
#: arrays, which is what ``np.asarray`` gives a plain list of ``str``.
_TEXT_KINDS = "USO"


def _declare(
    arrays: Mapping[str, np.ndarray], chunk_bytes: int
) -> tuple[list[np.ndarray | None], list[BufferSpec]]:
    """Validates arrays and describes them for a header.

    Shared by both upload streams, so an object upload and a bare data upload
    cannot disagree about what a well-formed declaration is.

    A text array is declared complete: its strings go in the header and its
    entry in the returned list is ``None``, because there is nothing left to
    chunk. The list still holds a slot for it so that a chunk's index keeps
    matching the header's.
    """
    if not arrays:
        raise ValueError("an upload needs at least one array")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    prepared: list[np.ndarray | None] = []
    specs: list[BufferSpec] = []
    for buffer_name, array in arrays.items():
        array = _wire_ready(np.asarray(array))
        if array.size == 0:
            raise ValueError(f"array {buffer_name!r} is empty")

        if array.dtype.kind in _TEXT_KINDS:
            prepared.append(None)
            specs.append(_text_spec(buffer_name, array))
            continue

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


def _text_spec(buffer_name: str, array: np.ndarray) -> BufferSpec:
    """Declares one text array, values and all.

    ``byte_length`` is 0 and no chunk ever follows: the strings travel inline,
    so the array is committed by the header alone. numpy's own width is
    dropped on the way — ``<U5`` pads every entry out to five characters and
    the wire carries each string at its own length.
    """
    values = [str(value) for value in array.reshape(-1)]
    if any("\x00" in value for value in values):
        raise ValueError(
            f"array {buffer_name!r} holds a string with an embedded null; "
            "the wire carries text, not packed bytes"
        )
    return BufferSpec(
        name=buffer_name,
        dtype=Dtype.DTYPE_STRING,
        shape=list(array.shape),
        byte_length=0,
        values=values,
    )


def _payload_chunks(
    prepared: list[np.ndarray | None], chunk_bytes: int
) -> Iterator[Chunk]:
    """Slices every array into wire-sized chunks, in declaration order.

    Text arrays are already on the wire in the header, so they are skipped —
    their index is still consumed, keeping every chunk pointed at the right
    declaration.
    """
    for index, array in enumerate(prepared):
        if array is None:
            continue
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

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
            info.buffer.name: info.handle.id for info in response.arrays
        }

    def list_data(self) -> list[DataSummary | GeometrySummary]:
        """Everything currently held, whether or not anything draws it.

        Arrays first, then the meshes filters have assembled. One handle space,
        so both come back from one call, and each entry says which it is by its
        own type.
        """
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
        # Cast op needed for strict type checking
        if isinstance(scale, (int, float)):
            scale = cast(tuple[float, float, float], (float(scale),) * 3)
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

    def delete_object(self, handle: int) -> tuple[int, ...]:
        """Removes one object, returning the handles actually removed.

        Deletes exactly what you name, and nothing else. Every child survives
        it, in the way that suits what it is: a child object becomes a root,
        since its transform is a place it still occupies; a child actor is
        detached and stops being drawn, since its transform is only an offset
        from the object it was under. Give a detached actor a new home with
        ``set_actor(handle, parent=...)``.

        No arrays are freed either; :meth:`release_data` does that, and
        :meth:`remove_actor` destroys an actor.

        Returns an empty tuple if the handle was already gone.
        """
        response = self._scene.DeleteObject(
            DeleteObjectRequest(handle=ObjectHandle(id=handle))
        )
        return tuple(h.id for h in response.removed)

    def add_actor(
        self,
        kind: str,
        *,
        parent: int | None = None,
        parents: Sequence[int] | None = None,
        params: Mapping[str, float | bool | str] | None = None,
        subset: np.ndarray | None = None,
        per_cell: bool = False,
    ) -> ActorSummary:
        """Draws something, under an object or under one made for it.

        Adds rather than replaces: an object may carry several actors at once,
        each configured on its own.

        ``parent`` is where it appears — whose transform it inherits. Leave it
        out and an object is created to hold this actor, named after its kind;
        the handle comes back as ``.parents[0]``. An actor has no place of its
        own, so it always ends up under something::

            data = client.upload_data({"xyz": positions})
            actor = client.add_actor("points",
                                     params={"positions": Bind(data["xyz"])})
            client.set_transform(actor.parents[0], translation=(10, 0, 0))

        ``parents`` draws this one actor under each of several objects at
        once. It stays one actor — one mesh, one set of settings — so changing
        it changes every copy::

            here = client.create_object("here")
            there = client.create_object("there")
            client.set_transform(there, translation=(10, 0, 0))
            actor = client.add_actor("points", parents=[here, there],
                                     params={"positions": Bind(data["xyz"])})
            client.set_actor(actor.handle, {"size": 0.2})   # both of them

        Two actors binding the same array is the other thing: two drawings that
        happen to look alike, each configured on its own.

        *What* it draws is in ``params``, as :class:`Bind` values against the
        inputs its kind declares.

        Parameters left out take the kind's default, not the value some other
        actor happens to have.

        ``subset`` draws only part of the bound data — a boolean mask over every
        element, or an integer array of the elements to keep. This is what
        makes several actors worth having: one structure shown as cartoon over
        its protein and ball-and-stick over its ligand is two actors with two
        subsets. Selections are computed here rather than described to the
        server, so anything numpy can express works::

            client.add_actor("surface", parent=node,
                             subset=positions[:, 2] > 0, ...)

        A mesh cell survives only when all of its corners do, and a bond only
        when both its atoms do, so a cut leaves a clean boundary rather than
        stretched or dangling geometry.
        """
        if parent is not None and parents is not None:
            raise ValueError("pass parent or parents, not both")
        request = AddActorRequest(kind=kind, params=_params(params))
        # One spelling on the wire. `parent` is the singular convenience, kept
        # because drawing in one place is much the commoner case.
        for handle in (parents if parents is not None else _maybe(parent)):
            request.parents.append(ObjectHandle(id=handle))
        if subset is not None:
            request.subset.CopyFrom(_subset(subset, per_cell=per_cell))
        return _actor(self._scene.AddActor(request).actor)

    def set_actor(
        self,
        handle: int,
        params: Mapping[str, float | bool | str] | None = None,
        *,
        visible: bool | None = None,
        subset: np.ndarray | None = None,
        per_cell: bool = False,
        clear_subset: bool = False,
        parent: int | None = None,
        parents: Sequence[int] | None = None,
    ) -> ActorSummary:
        """Changes an actor, leaving anything unnamed alone.

        Parameters are merged, so passing one setting keeps the rest — the
        opposite of :meth:`add_actor`, where an absent parameter takes its
        default.

        Out-of-range values are clamped rather than rejected, so a slider driven
        past its limit does not raise.

        Omitting ``subset`` leaves the selection alone; ``clear_subset=True``
        goes back to drawing the whole dataset. The two are separate because
        "unchanged" and "cleared" both have to be expressible.

        ``parent`` moves the actor under one object, and ``parents`` replaces
        the whole set of objects it is drawn under — which both adds an
        appearance and takes one away. ``parents=[]`` takes it off screen
        without removing it. Passing neither leaves the placements alone.

        This is how an actor that lost its last object is drawn again, and how
        one drawing is put in several places at once.
        """
        if subset is not None and clear_subset:
            raise ValueError("pass a subset or clear_subset, not both")

        request = SetActorRequest(
            handle=ActorHandle(id=handle),
            params=_params(params),
            clear_subset=clear_subset,
        )
        if visible is not None:
            request.visible = visible
        if subset is not None:
            request.subset.CopyFrom(_subset(subset, per_cell=per_cell))
        if parent is not None and parents is not None:
            raise ValueError("pass parent or parents, not both")
        if parent is not None or parents is not None:
            wanted = parents if parents is not None else _maybe(parent)
            request.parents.CopyFrom(
                ObjectHandles(handles=[ObjectHandle(id=handle) for handle in wanted])
            )
        return _actor(self._scene.SetActor(request).actor)

    def remove_actor(self, handle: int) -> bool:
        """Stops drawing something one way, leaving the object and its data.

        Returns False if the handle was already gone.
        """
        response = self._scene.RemoveActor(RemoveActorRequest(handle=ActorHandle(id=handle)))
        return response.removed

    def list_actors(self, parent: int | None = None) -> list[ActorSummary]:
        """Lists actors, optionally only those drawn under one object."""
        request = ListActorsRequest()
        if parent is not None:
            request.parent.CopyFrom(ObjectHandle(id=parent))
        response = self._scene.ListActors(request)
        return [_actor(info) for info in response.actors]

    def actor_kinds(self) -> dict[str, ActorKindSummary]:
        """The ways of drawing this server supports, keyed by kind id.

        Kinds come from the rendering pathway the server was built with, so ask
        rather than assuming: a hardcoded list here would eventually offer
        something that silently does nothing.

        A kind's ``params`` include its array inputs, each saying what element
        types and shape it takes and whether the kind can draw without it. That
        is what to bind an uploaded array to::

            wanted = client.actor_kinds()["points"]
            [p.id for p in wanted.params if p.type == "array" and p.required]
        """
        response = self._scene.ListActorKinds(ListActorKindsRequest())
        return {key: _kind(info) for key, info in response.kinds.items()}

    def add_filter(
        self,
        kind: str,
        *,
        params: Mapping[str, float | bool | str] | None = None,
    ) -> FilterSummary:
        """Derives arrays from arrays. Draws nothing.

        A filter is how derived data is made — a colour map over a field, a
        ribbon through a backbone, a surface out of a grid. What comes out are
        ordinary arrays with ordinary handles, so an actor binds them exactly as
        it binds an upload and cannot tell the two apart::

            field = client.upload_data({"b_factor": values})["b_factor"]
            colours = client.add_filter("colormap",
                                        params={"values": Bind(field)})
            client.add_actor("surface", params={
                "positions": Bind(xyz),
                "indices": Bind(tris),
                "colour": Bind(colours["colour"]),
            })

        That separation is the point: one generated result can feed several
        actors, so drawing a ribbon as a lit surface *and* as an absorbing medium
        builds it once rather than twice.

        The returned handles are usable immediately. Until the first run
        finishes they name empty arrays, and an actor bound to one draws nothing
        rather than failing.

        Parameters left out take the kind's default. Ask :meth:`filter_kinds`
        what a kind takes and what it produces.
        """
        request = AddFilterRequest(kind=kind, params=_params(params))
        return _filter(self._scene.AddFilter(request).filter)

    def set_filter(
        self,
        handle: int,
        params: Mapping[str, float | bool | str] | None = None,
    ) -> FilterSummary:
        """Changes a filter, leaving anything unnamed alone.

        Parameters are merged, the opposite of :meth:`add_filter` where an
        absent one takes its default. The output handles do not change, so
        everything bound to this filter picks up the new result on its own::

            client.set_filter(colours.handle, {"map": "cool-warm"})

        Raises ``FAILED_PRECONDITION`` if the binding would feed the filter its
        own output, directly or through others — such a graph could never come
        to rest.
        """
        request = SetFilterRequest(
            handle=FilterHandle(id=handle), params=_params(params)
        )
        return _filter(self._scene.SetFilter(request).filter)

    def remove_filter(self, handle: int) -> bool:
        """Removes a filter and forgets the arrays it was writing.

        This is the only way those handles go away: :meth:`release_data` refuses
        them one at a time, because releasing an array something is still
        generating leaves the filter producing into nothing.

        Returns False if the handle was already gone.
        """
        request = RemoveFilterRequest(handle=FilterHandle(id=handle))
        return self._scene.RemoveFilter(request).removed

    def filters(self) -> list[FilterSummary]:
        """Every filter in the scene, in handle order."""
        response = self._scene.ListFilters(ListFiltersRequest())
        return [_filter(info) for info in response.filters]

    def filter_kinds(self) -> dict[str, FilterKindSummary]:
        """The ways of deriving data this server supports, keyed by kind id.

        Ask rather than assuming, as with :meth:`actor_kinds`. A kind's
        ``outputs`` say what it writes before you have run it::

            colormap = client.filter_kinds()["colormap"]
            [(o.id, o.dtype, o.shape) for o in colormap.outputs]
        """
        response = self._scene.ListFilterKinds(ListFilterKindsRequest())
        return {key: _filter_kind(info) for key, info in response.kinds.items()}
