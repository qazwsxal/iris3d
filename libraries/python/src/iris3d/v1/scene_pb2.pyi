from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Dtype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[Dtype]
    DTYPE_UINT8: _ClassVar[Dtype]
    DTYPE_INT8: _ClassVar[Dtype]
    DTYPE_UINT16: _ClassVar[Dtype]
    DTYPE_INT16: _ClassVar[Dtype]
    DTYPE_UINT32: _ClassVar[Dtype]
    DTYPE_INT32: _ClassVar[Dtype]
    DTYPE_UINT64: _ClassVar[Dtype]
    DTYPE_INT64: _ClassVar[Dtype]
    DTYPE_FLOAT32: _ClassVar[Dtype]
    DTYPE_FLOAT64: _ClassVar[Dtype]
DTYPE_UNSPECIFIED: Dtype
DTYPE_UINT8: Dtype
DTYPE_INT8: Dtype
DTYPE_UINT16: Dtype
DTYPE_INT16: Dtype
DTYPE_UINT32: Dtype
DTYPE_INT32: Dtype
DTYPE_UINT64: Dtype
DTYPE_INT64: Dtype
DTYPE_FLOAT32: Dtype
DTYPE_FLOAT64: Dtype

class BufferSpec(_message.Message):
    __slots__ = ("name", "dtype", "shape", "byte_length")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BYTE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    name: str
    dtype: Dtype
    shape: _containers.RepeatedScalarFieldContainer[int]
    byte_length: int
    def __init__(self, name: _Optional[str] = ..., dtype: _Optional[_Union[Dtype, str]] = ..., shape: _Optional[_Iterable[int]] = ..., byte_length: _Optional[int] = ...) -> None: ...

class ObjectHeader(_message.Message):
    __slots__ = ("name", "buffers")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    buffers: _containers.RepeatedCompositeFieldContainer[BufferSpec]
    def __init__(self, name: _Optional[str] = ..., buffers: _Optional[_Iterable[_Union[BufferSpec, _Mapping]]] = ...) -> None: ...

class Chunk(_message.Message):
    __slots__ = ("buffer_index", "offset", "data")
    BUFFER_INDEX_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    buffer_index: int
    offset: int
    data: bytes
    def __init__(self, buffer_index: _Optional[int] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class UploadObjectRequest(_message.Message):
    __slots__ = ("header", "chunk")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    header: ObjectHeader
    chunk: Chunk
    def __init__(self, header: _Optional[_Union[ObjectHeader, _Mapping]] = ..., chunk: _Optional[_Union[Chunk, _Mapping]] = ...) -> None: ...

class UploadObjectResponse(_message.Message):
    __slots__ = ("handle", "total_bytes")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    total_bytes: int
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class ObjectHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class ObjectInfo(_message.Message):
    __slots__ = ("handle", "name", "buffers", "total_bytes", "dataset_kind", "representations", "parent")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATASET_KIND_FIELD_NUMBER: _ClassVar[int]
    REPRESENTATIONS_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    name: str
    buffers: _containers.RepeatedCompositeFieldContainer[BufferSpec]
    total_bytes: int
    dataset_kind: str
    representations: _containers.RepeatedScalarFieldContainer[str]
    parent: ObjectHandle
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., name: _Optional[str] = ..., buffers: _Optional[_Iterable[_Union[BufferSpec, _Mapping]]] = ..., total_bytes: _Optional[int] = ..., dataset_kind: _Optional[str] = ..., representations: _Optional[_Iterable[str]] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class ListObjectsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListObjectsResponse(_message.Message):
    __slots__ = ("objects",)
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    objects: _containers.RepeatedCompositeFieldContainer[ObjectInfo]
    def __init__(self, objects: _Optional[_Iterable[_Union[ObjectInfo, _Mapping]]] = ...) -> None: ...

class CreateObjectRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class CreateObjectResponse(_message.Message):
    __slots__ = ("handle",)
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class SetParentRequest(_message.Message):
    __slots__ = ("handle", "parent", "keep_world_transform")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    KEEP_WORLD_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    parent: ObjectHandle
    keep_world_transform: bool
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ..., keep_world_transform: _Optional[bool] = ...) -> None: ...

class SetParentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Vector3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Quaternion(_message.Message):
    __slots__ = ("x", "y", "z", "w")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    w: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., w: _Optional[float] = ...) -> None: ...

class SetTransformRequest(_message.Message):
    __slots__ = ("handle", "translation", "rotation", "scale")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    translation: Vector3
    rotation: Quaternion
    scale: Vector3
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., translation: _Optional[_Union[Vector3, _Mapping]] = ..., rotation: _Optional[_Union[Quaternion, _Mapping]] = ..., scale: _Optional[_Union[Vector3, _Mapping]] = ...) -> None: ...

class SetTransformResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteObjectRequest(_message.Message):
    __slots__ = ("handle", "recursive")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    RECURSIVE_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    recursive: bool
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., recursive: _Optional[bool] = ...) -> None: ...

class DeleteObjectResponse(_message.Message):
    __slots__ = ("deleted", "removed")
    DELETED_FIELD_NUMBER: _ClassVar[int]
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    removed: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    def __init__(self, deleted: _Optional[bool] = ..., removed: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ...) -> None: ...
