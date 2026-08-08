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
    BufferSpec,
    Chunk,
    CreateObjectRequest,
    DeleteObjectRequest,
    Dtype,
    ListObjectsRequest,
    ObjectHandle,
    ObjectHeader,
    ObjectInfo,
    Quaternion,
    SetParentRequest,
    SetTransformRequest,
    UploadObjectRequest,
    Vector3,
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
    #: Ways the object is currently drawn; empty if nothing draws it.
    representations: tuple[str, ...]
    #: Parent handle in the scene tree, or None for a root object.
    parent: int | None


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
        representations=tuple(info.representations),
        parent=info.parent.id if info.HasField("parent") else None,
    )


def upload_messages(
    name: str,
    arrays: Mapping[str, np.ndarray],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[UploadObjectRequest]:
    """Builds the request stream for one object.

    Yields a header declaring every buffer, then the buffers' bytes in order.
    Exposed separately from :meth:`Client.upload_object` so callers can wrap or
    inspect the stream — for progress reporting, say.
    """
    if not arrays:
        raise ValueError("an object needs at least one buffer")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    prepared = []
    specs = []
    for buffer_name, array in arrays.items():
        array = _wire_ready(np.asarray(array))
        if array.size == 0:
            raise ValueError(f"buffer {buffer_name!r} is empty")
        prepared.append(array)
        specs.append(
            BufferSpec(
                name=buffer_name,
                dtype=to_proto_dtype(array.dtype),
                shape=list(array.shape),
                byte_length=array.nbytes,
            )
        )

    yield UploadObjectRequest(header=ObjectHeader(name=name, buffers=specs))

    for index, array in enumerate(prepared):
        payload = memoryview(array).cast("B")
        for offset in range(0, len(payload), chunk_bytes):
            yield UploadObjectRequest(
                chunk=Chunk(
                    buffer_index=index,
                    offset=offset,
                    data=bytes(payload[offset : offset + chunk_bytes]),
                )
            )


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
    ) -> int:
        """Uploads one object and returns its handle.

        ``arrays`` maps buffer names to arrays. Conventional names are
        ``positions`` (float32, ``[n, 3]``), ``colors`` (uint8, ``[n, 3]`` or
        ``[n, 4]``), ``indices`` (uint32, ``[m, 3]``) and ``normals``
        (float32, ``[n, 3]``).
        """
        response = self._scene.UploadObject(upload_messages(name, arrays, chunk_bytes))
        return response.handle.id

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
