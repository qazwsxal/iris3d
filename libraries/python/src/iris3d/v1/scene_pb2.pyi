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
    DTYPE_STRING: _ClassVar[Dtype]
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
DTYPE_STRING: Dtype

class BufferSpec(_message.Message):
    __slots__ = ("name", "dtype", "shape", "byte_length", "values")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BYTE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    dtype: Dtype
    shape: _containers.RepeatedScalarFieldContainer[int]
    byte_length: int
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., dtype: _Optional[_Union[Dtype, str]] = ..., shape: _Optional[_Iterable[int]] = ..., byte_length: _Optional[int] = ..., values: _Optional[_Iterable[str]] = ...) -> None: ...

class Chunk(_message.Message):
    __slots__ = ("buffer_index", "offset", "data")
    BUFFER_INDEX_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    buffer_index: int
    offset: int
    data: bytes
    def __init__(self, buffer_index: _Optional[int] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class DataHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class UploadDataRequest(_message.Message):
    __slots__ = ("header", "chunk")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    header: DataHeader
    chunk: Chunk
    def __init__(self, header: _Optional[_Union[DataHeader, _Mapping]] = ..., chunk: _Optional[_Union[Chunk, _Mapping]] = ...) -> None: ...

class DataHeader(_message.Message):
    __slots__ = ("arrays",)
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[BufferSpec]
    def __init__(self, arrays: _Optional[_Iterable[_Union[BufferSpec, _Mapping]]] = ...) -> None: ...

class UploadDataResponse(_message.Message):
    __slots__ = ("arrays", "total_bytes")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[DataInfo]
    total_bytes: int
    def __init__(self, arrays: _Optional[_Iterable[_Union[DataInfo, _Mapping]]] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class DataInfo(_message.Message):
    __slots__ = ("handle", "spec")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    handle: DataHandle
    spec: BufferSpec
    def __init__(self, handle: _Optional[_Union[DataHandle, _Mapping]] = ..., spec: _Optional[_Union[BufferSpec, _Mapping]] = ...) -> None: ...

class ListDataRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDataResponse(_message.Message):
    __slots__ = ("arrays",)
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[DataInfo]
    def __init__(self, arrays: _Optional[_Iterable[_Union[DataInfo, _Mapping]]] = ...) -> None: ...

class ReleaseDataRequest(_message.Message):
    __slots__ = ("handles",)
    HANDLES_FIELD_NUMBER: _ClassVar[int]
    handles: _containers.RepeatedCompositeFieldContainer[DataHandle]
    def __init__(self, handles: _Optional[_Iterable[_Union[DataHandle, _Mapping]]] = ...) -> None: ...

class ReleaseDataResponse(_message.Message):
    __slots__ = ("released",)
    RELEASED_FIELD_NUMBER: _ClassVar[int]
    released: _containers.RepeatedCompositeFieldContainer[DataHandle]
    def __init__(self, released: _Optional[_Iterable[_Union[DataHandle, _Mapping]]] = ...) -> None: ...

class ObjectHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class ObjectInfo(_message.Message):
    __slots__ = ("handle", "name", "actors", "parent")
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    name: str
    actors: _containers.RepeatedCompositeFieldContainer[ActorInfo]
    parent: ObjectHandle
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ..., name: _Optional[str] = ..., actors: _Optional[_Iterable[_Union[ActorInfo, _Mapping]]] = ..., parent: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class ActorHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class ParamValue(_message.Message):
    __slots__ = ("number", "flag", "text", "vector", "data")
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    FLAG_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    number: float
    flag: bool
    text: str
    vector: VectorValue
    data: DataHandle
    def __init__(self, number: _Optional[float] = ..., flag: _Optional[bool] = ..., text: _Optional[str] = ..., vector: _Optional[_Union[VectorValue, _Mapping]] = ..., data: _Optional[_Union[DataHandle, _Mapping]] = ...) -> None: ...

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

class ChoiceParam(_message.Message):
    __slots__ = ("options", "default_value")
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedScalarFieldContainer[str]
    default_value: str
    def __init__(self, options: _Optional[_Iterable[str]] = ..., default_value: _Optional[str] = ...) -> None: ...

class ParamSpec(_message.Message):
    __slots__ = ("id", "label", "number", "flag", "choice", "array", "vector")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    FLAG_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    number: FloatParam
    flag: BoolParam
    choice: ChoiceParam
    array: ArrayParam
    vector: VectorParam
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., number: _Optional[_Union[FloatParam, _Mapping]] = ..., flag: _Optional[_Union[BoolParam, _Mapping]] = ..., choice: _Optional[_Union[ChoiceParam, _Mapping]] = ..., array: _Optional[_Union[ArrayParam, _Mapping]] = ..., vector: _Optional[_Union[VectorParam, _Mapping]] = ...) -> None: ...

class VectorValue(_message.Message):
    __slots__ = ("components",)
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    components: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, components: _Optional[_Iterable[float]] = ...) -> None: ...

class VectorParam(_message.Message):
    __slots__ = ("components", "default_value", "min", "max", "integral")
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    INTEGRAL_FIELD_NUMBER: _ClassVar[int]
    components: int
    default_value: _containers.RepeatedScalarFieldContainer[float]
    min: float
    max: float
    integral: bool
    def __init__(self, components: _Optional[int] = ..., default_value: _Optional[_Iterable[float]] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., integral: _Optional[bool] = ...) -> None: ...

class ArrayParam(_message.Message):
    __slots__ = ("dtypes", "shape", "required")
    DTYPES_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    dtypes: _containers.RepeatedScalarFieldContainer[Dtype]
    shape: _containers.RepeatedScalarFieldContainer[int]
    required: bool
    def __init__(self, dtypes: _Optional[_Iterable[_Union[Dtype, str]]] = ..., shape: _Optional[_Iterable[int]] = ..., required: _Optional[bool] = ...) -> None: ...

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
    __slots__ = ("map", "range", "flat")
    MAP_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    FLAT_FIELD_NUMBER: _ClassVar[int]
    map: str
    range: Range
    flat: Color
    def __init__(self, map: _Optional[str] = ..., range: _Optional[_Union[Range, _Mapping]] = ..., flat: _Optional[_Union[Color, _Mapping]] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("low", "high")
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    low: float
    high: float
    def __init__(self, low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...

class Subset(_message.Message):
    __slots__ = ("data", "dtype", "encoding", "association")
    class Encoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ENCODING_UNSPECIFIED: _ClassVar[Subset.Encoding]
        ENCODING_INDICES: _ClassVar[Subset.Encoding]
        ENCODING_MASK: _ClassVar[Subset.Encoding]
    ENCODING_UNSPECIFIED: Subset.Encoding
    ENCODING_INDICES: Subset.Encoding
    ENCODING_MASK: Subset.Encoding
    class Association(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ASSOCIATION_UNSPECIFIED: _ClassVar[Subset.Association]
        ASSOCIATION_PER_POINT: _ClassVar[Subset.Association]
        ASSOCIATION_PER_CELL: _ClassVar[Subset.Association]
    ASSOCIATION_UNSPECIFIED: Subset.Association
    ASSOCIATION_PER_POINT: Subset.Association
    ASSOCIATION_PER_CELL: Subset.Association
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    ASSOCIATION_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dtype: Dtype
    encoding: Subset.Encoding
    association: Subset.Association
    def __init__(self, data: _Optional[bytes] = ..., dtype: _Optional[_Union[Dtype, str]] = ..., encoding: _Optional[_Union[Subset.Encoding, str]] = ..., association: _Optional[_Union[Subset.Association, str]] = ...) -> None: ...

class ActorInfo(_message.Message):
    __slots__ = ("handle", "kind", "params", "color", "visible", "subset", "parents")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    SUBSET_FIELD_NUMBER: _ClassVar[int]
    PARENTS_FIELD_NUMBER: _ClassVar[int]
    handle: ActorHandle
    kind: str
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    visible: bool
    subset: SubsetInfo
    parents: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    def __init__(self, handle: _Optional[_Union[ActorHandle, _Mapping]] = ..., kind: _Optional[str] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ..., visible: _Optional[bool] = ..., subset: _Optional[_Union[SubsetInfo, _Mapping]] = ..., parents: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ...) -> None: ...

class SubsetInfo(_message.Message):
    __slots__ = ("encoding", "association", "selected")
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    ASSOCIATION_FIELD_NUMBER: _ClassVar[int]
    SELECTED_FIELD_NUMBER: _ClassVar[int]
    encoding: Subset.Encoding
    association: Subset.Association
    selected: int
    def __init__(self, encoding: _Optional[_Union[Subset.Encoding, str]] = ..., association: _Optional[_Union[Subset.Association, str]] = ..., selected: _Optional[int] = ...) -> None: ...

class ActorKindInfo(_message.Message):
    __slots__ = ("id", "label", "params")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    params: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., params: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ...) -> None: ...

class AddActorRequest(_message.Message):
    __slots__ = ("kind", "params", "color", "subset", "parents")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    SUBSET_FIELD_NUMBER: _ClassVar[int]
    PARENTS_FIELD_NUMBER: _ClassVar[int]
    kind: str
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    subset: Subset
    parents: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    def __init__(self, kind: _Optional[str] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ..., subset: _Optional[_Union[Subset, _Mapping]] = ..., parents: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ...) -> None: ...

class AddActorResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ActorInfo
    def __init__(self, actor: _Optional[_Union[ActorInfo, _Mapping]] = ...) -> None: ...

class SetActorRequest(_message.Message):
    __slots__ = ("handle", "params", "color", "visible", "subset", "clear_subset", "parents")
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
    SUBSET_FIELD_NUMBER: _ClassVar[int]
    CLEAR_SUBSET_FIELD_NUMBER: _ClassVar[int]
    PARENTS_FIELD_NUMBER: _ClassVar[int]
    handle: ActorHandle
    params: _containers.MessageMap[str, ParamValue]
    color: ColorSpec
    visible: bool
    subset: Subset
    clear_subset: bool
    parents: ObjectHandles
    def __init__(self, handle: _Optional[_Union[ActorHandle, _Mapping]] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., color: _Optional[_Union[ColorSpec, _Mapping]] = ..., visible: _Optional[bool] = ..., subset: _Optional[_Union[Subset, _Mapping]] = ..., clear_subset: _Optional[bool] = ..., parents: _Optional[_Union[ObjectHandles, _Mapping]] = ...) -> None: ...

class ObjectHandles(_message.Message):
    __slots__ = ("handles",)
    HANDLES_FIELD_NUMBER: _ClassVar[int]
    handles: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    def __init__(self, handles: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ...) -> None: ...

class SetActorResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ActorInfo
    def __init__(self, actor: _Optional[_Union[ActorInfo, _Mapping]] = ...) -> None: ...

class RemoveActorRequest(_message.Message):
    __slots__ = ("handle",)
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    handle: ActorHandle
    def __init__(self, handle: _Optional[_Union[ActorHandle, _Mapping]] = ...) -> None: ...

class RemoveActorResponse(_message.Message):
    __slots__ = ("removed",)
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    removed: bool
    def __init__(self, removed: _Optional[bool] = ...) -> None: ...

class ListActorsRequest(_message.Message):
    __slots__ = ("parent",)
    PARENT_FIELD_NUMBER: _ClassVar[int]
    parent: ObjectHandle
    def __init__(self, parent: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class ListActorsResponse(_message.Message):
    __slots__ = ("actors",)
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    actors: _containers.RepeatedCompositeFieldContainer[ActorInfo]
    def __init__(self, actors: _Optional[_Iterable[_Union[ActorInfo, _Mapping]]] = ...) -> None: ...

class FilterHandle(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class FilterInfo(_message.Message):
    __slots__ = ("handle", "kind", "params", "outputs")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    handle: FilterHandle
    kind: str
    params: _containers.MessageMap[str, ParamValue]
    outputs: _containers.RepeatedCompositeFieldContainer[FilterOutput]
    def __init__(self, handle: _Optional[_Union[FilterHandle, _Mapping]] = ..., kind: _Optional[str] = ..., params: _Optional[_Mapping[str, ParamValue]] = ..., outputs: _Optional[_Iterable[_Union[FilterOutput, _Mapping]]] = ...) -> None: ...

class FilterOutput(_message.Message):
    __slots__ = ("id", "handle")
    ID_FIELD_NUMBER: _ClassVar[int]
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    id: str
    handle: DataHandle
    def __init__(self, id: _Optional[str] = ..., handle: _Optional[_Union[DataHandle, _Mapping]] = ...) -> None: ...

class FilterKindInfo(_message.Message):
    __slots__ = ("id", "label", "params", "outputs")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    params: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    outputs: _containers.RepeatedCompositeFieldContainer[OutputSpec]
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., params: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[OutputSpec, _Mapping]]] = ...) -> None: ...

class OutputSpec(_message.Message):
    __slots__ = ("id", "label", "dtype", "shape")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    dtype: Dtype
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., dtype: _Optional[_Union[Dtype, str]] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class AddFilterRequest(_message.Message):
    __slots__ = ("kind", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    KIND_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    kind: str
    params: _containers.MessageMap[str, ParamValue]
    def __init__(self, kind: _Optional[str] = ..., params: _Optional[_Mapping[str, ParamValue]] = ...) -> None: ...

class AddFilterResponse(_message.Message):
    __slots__ = ("filter",)
    FILTER_FIELD_NUMBER: _ClassVar[int]
    filter: FilterInfo
    def __init__(self, filter: _Optional[_Union[FilterInfo, _Mapping]] = ...) -> None: ...

class SetFilterRequest(_message.Message):
    __slots__ = ("handle", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ParamValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ParamValue, _Mapping]] = ...) -> None: ...
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    handle: FilterHandle
    params: _containers.MessageMap[str, ParamValue]
    def __init__(self, handle: _Optional[_Union[FilterHandle, _Mapping]] = ..., params: _Optional[_Mapping[str, ParamValue]] = ...) -> None: ...

class SetFilterResponse(_message.Message):
    __slots__ = ("filter",)
    FILTER_FIELD_NUMBER: _ClassVar[int]
    filter: FilterInfo
    def __init__(self, filter: _Optional[_Union[FilterInfo, _Mapping]] = ...) -> None: ...

class RemoveFilterRequest(_message.Message):
    __slots__ = ("handle",)
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    handle: FilterHandle
    def __init__(self, handle: _Optional[_Union[FilterHandle, _Mapping]] = ...) -> None: ...

class RemoveFilterResponse(_message.Message):
    __slots__ = ("removed",)
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    removed: bool
    def __init__(self, removed: _Optional[bool] = ...) -> None: ...

class ListFiltersRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListFiltersResponse(_message.Message):
    __slots__ = ("filters",)
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    filters: _containers.RepeatedCompositeFieldContainer[FilterInfo]
    def __init__(self, filters: _Optional[_Iterable[_Union[FilterInfo, _Mapping]]] = ...) -> None: ...

class ListFilterKindsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListFilterKindsResponse(_message.Message):
    __slots__ = ("kinds",)
    class KindsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: FilterKindInfo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[FilterKindInfo, _Mapping]] = ...) -> None: ...
    KINDS_FIELD_NUMBER: _ClassVar[int]
    kinds: _containers.MessageMap[str, FilterKindInfo]
    def __init__(self, kinds: _Optional[_Mapping[str, FilterKindInfo]] = ...) -> None: ...

class ListActorKindsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListActorKindsResponse(_message.Message):
    __slots__ = ("kinds",)
    class KindsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ActorKindInfo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ActorKindInfo, _Mapping]] = ...) -> None: ...
    KINDS_FIELD_NUMBER: _ClassVar[int]
    kinds: _containers.MessageMap[str, ActorKindInfo]
    def __init__(self, kinds: _Optional[_Mapping[str, ActorKindInfo]] = ...) -> None: ...

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
    __slots__ = ("handle",)
    HANDLE_FIELD_NUMBER: _ClassVar[int]
    handle: ObjectHandle
    def __init__(self, handle: _Optional[_Union[ObjectHandle, _Mapping]] = ...) -> None: ...

class DeleteObjectResponse(_message.Message):
    __slots__ = ("deleted", "removed")
    DELETED_FIELD_NUMBER: _ClassVar[int]
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    removed: _containers.RepeatedCompositeFieldContainer[ObjectHandle]
    def __init__(self, deleted: _Optional[bool] = ..., removed: _Optional[_Iterable[_Union[ObjectHandle, _Mapping]]] = ...) -> None: ...
