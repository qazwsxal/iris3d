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
    __slots__ = ("handle", "name", "buffers", "total_bytes", "dataset_kind", "parent", "drawn_by")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATASET_KIND_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    DRAWN_BY_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    name: str
    buffers: _containers.RepeatedCompositeFieldContainer[BufferSpec]
    total_bytes: int
    dataset_kind: str
    parent: ObjectHandle
    drawn_by: _containers.RepeatedCompositeFieldContainer[RepresentationInfo]
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., name: _Optional[str] = ..., buffers: _Optional[_Iterable[_Union[BufferSpec, _Mapping]]] = ..., total_bytes: _Optional[int] = ..., dataset_kind: _Optional[str] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ..., drawn_by: _Optional[_Iterable[_Union[RepresentationInfo, _Mapping]]] = ...) -> None: ...

class RepresentationHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class ParamValue(_message.Message):
    __slots__ = ("number", "flag")
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    FLAG_FIELD_NUMBER: _ClassVar[int]
    number: float
    flag: bool
    def __init__(self, number: _Optional[float] = ..., flag: _Optional[bool] = ...) -> None: ...

class FloatParam(_message.Message):
    __slots__ = ("default_value", "min", "max", "logarithmic")
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    LOGARITHMIC_FIELD_NUMBER: _ClassVar[int]
    default_value: float
    min: float
    max: float
    logarithmic: bool
    def __init__(self, default_value: _Optional[float] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., logarithmic: _Optional[bool] = ...) -> None: ...

class BoolParam(_message.Message):
    __slots__ = ("default_value",)
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    default_value: bool
    def __init__(self, default_value: _Optional[bool] = ...) -> None: ...

class ParamSpec(_message.Message):
    __slots__ = ("id", "label", "number", "flag")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    FLAG_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    number: FloatParam
    flag: BoolParam
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., number: _Optional[_Union[FloatParam, _Mapping]] = ..., flag: _Optional[_Union[BoolParam, _Mapping]] = ...) -> None: ...

class Color(_message.Message):
    __slots__ = ("r", "g", "b")
    R_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    r: float
    g: float
    b: float
    def __init__(self, r: _Optional[float] = ..., g: _Optional[float] = ..., b: _Optional[float] = ...) -> None: ...

class ColorSpec(_message.Message):
    __slots__ = ("field", "map", "range", "flat")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    FLAT_FIELD_NUMBER: _ClassVar[int]
    field: str
    map: str
    range: Range
    flat: Color
    def __init__(self, field: _Optional[str] = ..., map: _Optional[str] = ..., range: _Optional[_Union[Range, _Mapping]] = ..., flat: _Optional[_Union[Color, _Mapping]] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("low", "high")
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    low: float
    high: float
    def __init__(self, low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...

class RepresentationInfo(_message.Message):
    __slots__ = ("handle", "kind", "source", "parent", "params", "color", "visible")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    handle: RepresentationHandle
    kind: str
    source: ObjectHandle
    parent: ObjectHandle
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    visible: bool
    def __init__(self, handle: _Optional[_Union[RepresentationHandle, _Mapping]] = ..., kind: _Optional[str] = ..., source: _Optional[_Union[ObjectHandle, _Mapping]] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ..., visible: _Optional[bool] = ...) -> None: ...

class RepresentationKindInfo(_message.Message):
    __slots__ = ("id", "label", "supports", "params")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    supports: _containers.RepeatedScalarFieldContainer[str]
    params: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., supports: _Optional[_Iterable[str]] = ..., params: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ...) -> None: ...

class AddRepresentationRequest(_message.Message):
    __slots__ = ("source", "kind", "parent", "params", "color")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    source: ObjectHandle
    kind: str
    parent: ObjectHandle
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    def __init__(self, source: _Optional[_Union[ObjectHandle, _Mapping]] = ..., kind: _Optional[str] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ...) -> None: ...

class AddRepresentationResponse(_message.Message):
    __slots__ = ("representation",)
    REPRESENTATION_FIELD_NUMBER: _ClassVar[int]
    representation: RepresentationInfo
    def __init__(self, representation: _Optional[_Union[RepresentationInfo, _Mapping]] = ...) -> None: ...

class SetRepresentationRequest(_message.Message):
    __slots__ = ("handle", "params", "color", "visible")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    handle: RepresentationHandle
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    visible: bool
    def __init__(self, handle: _Optional[_Union[RepresentationHandle, _Mapping]] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ..., visible: _Optional[bool] = ...) -> None: ...

class SetRepresentationResponse(_message.Message):
    __slots__ = ("representation",)
    REPRESENTATION_FIELD_NUMBER: _ClassVar[int]
    representation: RepresentationInfo
    def __init__(self, representation: _Optional[_Union[RepresentationInfo, _Mapping]] = ...) -> None: ...

class RemoveRepresentationRequest(_message.Message):
    __slots__ = ("handle",)
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    handle: RepresentationHandle
    def __init__(self, handle: _Optional[_Union[RepresentationHandle, _Mapping]] = ...) -> None: ...

class RemoveRepresentationResponse(_message.Message):
    __slots__ = ("removed",)
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    removed: bool
    def __init__(self, removed: _Optional[bool] = ...) -> None: ...

class ListRepresentationsRequest(_message.Message):
    __slots__ = ("source",)
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    source: ObjectHandle
    def __init__(self, source: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class ListRepresentationsResponse(_message.Message):
    __slots__ = ("representations",)
    REPRESENTATIONS_FIELD_NUMBER: _ClassVar[int]
    representations: _containers.RepeatedCompositeFieldContainer[RepresentationInfo]
    def __init__(self, representations: _Optional[_Iterable[_Union[RepresentationInfo, _Mapping]]] = ...) -> None: ...

class ListRepresentationKindsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListRepresentationKindsResponse(_message.Message):
    __slots__ = ("kinds",)
    KINDS_FIELD_NUMBER: _ClassVar[int]
    kinds: _containers.RepeatedCompositeFieldContainer[RepresentationKindInfo]
    def __init__(self, kinds: _Optional[_Iterable[_Union[RepresentationKindInfo, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("deleted", "removed", "removed_representations")
    DELETED_FIELD_NUMBER: _ClassVar[int]
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    REMOVED_REPRESENTATIONS_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    removed: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    removed_representations: _containers.RepeatedCompositeFieldContainer[RepresentationHandle]
    def __init__(self, deleted: _Optional[bool] = ..., removed: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ..., removed_representations: _Optional[_Iterable[_Union[RepresentationHandle, _Mapping]]] = ...) -> None: ...
