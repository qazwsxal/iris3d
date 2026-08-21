//! Between the wire types and the scene's own.
//!
//! Every function here is one direction of one conversion. They are together
//! because they are all the same kind of tedium and none of them decides
//! anything: the service methods stay readable by having no conversion inline,
//! and a change to the proto lands in one file.

use tonic::Status;

use crate::filter::{FilterKindSummary, FilterSummary, OutputKind};
use crate::model::{ParamKind, ParamMap, ParamSpec, ParamValue, SceneError};
use crate::scene::{
    ActorSummary, BufferMeta, DataSummary, Dtype, HeldMeta, KindSummary, ObjectSummary,
};

use super::proto::{
    ActorHandle, ActorInfo, ActorKindInfo, ArrayOutput, ArrayParam, BoolParam, BufferSpec,
    ChoiceParam, DataHandle, DataInfo, Dtype as ProtoDtype, FilterHandle, FilterInfo,
    FilterKindInfo, FilterOutput, FloatParam, GeometryOutput, GeometryParam, GeometrySpec,
    ObjectHandle, ObjectInfo, OutputSpec as ProtoOutputSpec, ParamSpec as ProtoSpec,
    ParamValue as ProtoParam, TextParam, VectorParam, VectorValue, data_info, output_spec,
    param_spec, param_value::Value,
};

pub(crate) fn scene_error(error: SceneError) -> Status {
    match error {
        SceneError::NoSuchObject(_)
        | SceneError::NoSuchActor(_)
        | SceneError::NoSuchData(_)
        | SceneError::NoSuchFilter(_) => Status::not_found(error.to_string()),
        // The caller's own request is at fault, and the declaration it needed in
        // order to get it right is in ListActorKinds.
        SceneError::MissingInput { .. } | SceneError::BadBinding { .. } => {
            Status::invalid_argument(error.to_string())
        }
        // The caller named something that does not exist in this build, which
        // it could have discovered with ListActorKinds.
        SceneError::UnknownKind { .. } | SceneError::UnknownFilterKind { .. } => {
            Status::invalid_argument(error.to_string())
        }
        // The request was well-formed but the scene is not in a state where it
        // can be honoured. Both cycles land here for the same reason: the call
        // is legible, and it is the *existing* graph that makes it impossible.
        SceneError::WouldCycle { .. }
        | SceneError::FilterCycle { .. }
        | SceneError::StillGenerated { .. } => Status::failed_precondition(error.to_string()),
    }
}

pub(crate) fn params_from_proto(
    params: std::collections::HashMap<String, ProtoParam>,
) -> Result<ParamMap, Status> {
    params
        .into_iter()
        .map(|(key, value)| {
            let value = match value.value {
                Some(Value::Number(number)) => ParamValue::Float(number as f32),
                Some(Value::Flag(flag)) => ParamValue::Bool(flag),
                Some(Value::Text(text)) => ParamValue::Text(text),
                Some(Value::Data(handle)) => ParamValue::Data(handle.id),
                Some(Value::Vector(vector)) => ParamValue::Vector(vector.components),
                Some(Value::Unset(_)) => ParamValue::Unset,
                // An empty `oneof` says nothing at all, and guessing which
                // parameter was meant is worse than saying so.
                None => {
                    return Err(Status::invalid_argument(format!(
                        "parameter \"{key}\" has no value set"
                    )));
                }
            };
            Ok((key, value))
        })
        .collect()
}

pub(crate) fn params_to_proto(params: &ParamMap) -> std::collections::HashMap<String, ProtoParam> {
    params
        .iter()
        .map(|(key, value)| {
            let value = match value {
                ParamValue::Float(number) => Value::Number(*number as f64),
                ParamValue::Bool(flag) => Value::Flag(*flag),
                ParamValue::Text(text) => Value::Text(text.clone()),
                ParamValue::Data(id) => Value::Data(DataHandle { id: *id }),
                ParamValue::Vector(values) => Value::Vector(VectorValue {
                    components: values.clone(),
                }),
                // Only ever travels *inbound*, as an instruction. A stored map
                // says what a thing is set to, and "cleared" is spelled there by
                // the key being absent — so this is unreachable from a real map
                // and would be a lie if it were not.
                ParamValue::Unset => Value::Unset(crate::grpc::proto::Unset {}),
            };
            (key.clone(), ProtoParam { value: Some(value) })
        })
        .collect()
}

pub(crate) fn actor_info(summary: &ActorSummary) -> ActorInfo {
    ActorInfo {
        handle: Some(ActorHandle { id: summary.id }),
        kind: summary.kind.clone(),

        parents: summary
            .parents
            .iter()
            .map(|id| ObjectHandle { id: *id })
            .collect(),
        params: params_to_proto(&summary.params),
        visible: summary.visible,
    }
}

/// One declared parameter, for the wire.
///
/// Shared by actor kinds and filter kinds, which declare their settings and
/// their array inputs with the same [`ParamSpec`] — so a client that can read
/// one listing can read the other.
pub(crate) fn spec_to_proto(spec: &ParamSpec) -> ProtoSpec {
    ProtoSpec {
        id: spec.id.to_string(),
        label: spec.label.to_string(),
        kind: Some(match spec.kind {
            ParamKind::Float {
                default,
                min,
                max,
                logarithmic,
            } => param_spec::Kind::Number(FloatParam {
                default_value: default as f64,
                min: min as f64,
                max: max as f64,
                logarithmic,
            }),
            ParamKind::Bool { default } => param_spec::Kind::Flag(BoolParam {
                default_value: default,
            }),
            ParamKind::Choice { options, default } => param_spec::Kind::Choice(ChoiceParam {
                options: options.iter().map(|option| option.to_string()).collect(),
                default_value: default.to_string(),
            }),
            ParamKind::Text { default } => param_spec::Kind::Text(TextParam {
                default_value: default.to_string(),
            }),
            ParamKind::Vector {
                components,
                default,
                min,
                max,
                integral,
            } => param_spec::Kind::Vector(VectorParam {
                components: components as u32,
                default_value: default.to_vec(),
                min,
                max,
                integral,
            }),
            // `structural` is not on the wire. It says whether new data here
            // forces a rebuild or only a repaint, which is the server scheduling
            // its own work — there is nothing a caller would do differently.
            ParamKind::Array {
                dtypes,
                shape,
                required,
                ..
            } => param_spec::Kind::Array(ArrayParam {
                dtypes: dtypes
                    .iter()
                    .map(|dtype| proto_dtype(*dtype) as i32)
                    .collect(),
                shape: shape.to_vec(),
                required,
            }),
            ParamKind::Geometry { required } => {
                param_spec::Kind::Geometry(GeometryParam { required })
            }
        }),
    }
}

pub(crate) fn kind_info(summary: &KindSummary) -> ActorKindInfo {
    ActorKindInfo {
        id: summary.id.clone(),
        label: summary.label.clone(),
        params: summary.params.iter().map(spec_to_proto).collect(),
    }
}

pub(crate) fn filter_info(summary: &FilterSummary) -> FilterInfo {
    FilterInfo {
        handle: Some(FilterHandle { id: summary.id }),
        kind: summary.kind.clone(),
        params: params_to_proto(&summary.params),
        outputs: summary
            .outputs
            .iter()
            .map(|(id, handle)| FilterOutput {
                id: id.clone(),
                handle: Some(DataHandle { id: *handle }),
            })
            .collect(),
        problem: summary.problem.clone(),
    }
}

pub(crate) fn filter_kind_info(summary: &FilterKindSummary) -> FilterKindInfo {
    FilterKindInfo {
        id: summary.id.clone(),
        label: summary.label.clone(),
        params: summary.params.iter().map(spec_to_proto).collect(),
        outputs: summary
            .outputs
            .iter()
            .map(|spec| ProtoOutputSpec {
                id: spec.id.to_string(),
                label: spec.label.to_string(),
                kind: Some(match spec.kind {
                    OutputKind::Array { dtype, shape } => output_spec::Kind::Array(ArrayOutput {
                        // `DTYPE_UNSPECIFIED` for an output whose type the run
                        // decides — the enum's zero value already means exactly
                        // "not stated", so nothing new is needed to say it.
                        dtype: dtype.map_or(0, |dtype| proto_dtype(dtype) as i32),
                        shape: shape.to_vec(),
                    }),
                    OutputKind::Geometry => output_spec::Kind::Geometry(GeometryOutput {}),
                }),
            })
            .collect(),
    }
}

pub(crate) fn object_info(summary: &ObjectSummary) -> ObjectInfo {
    ObjectInfo {
        handle: Some(ObjectHandle { id: summary.id }),
        name: summary.name.clone(),
        actors: summary.actors.iter().map(actor_info).collect(),
        parent: summary.parent.map(|id| ObjectHandle { id }),
    }
}

pub(crate) fn data_info(held: &DataSummary) -> DataInfo {
    DataInfo {
        handle: Some(DataHandle { id: held.id }),
        spec: Some(match &held.meta {
            HeldMeta::Array(meta) => data_info::Spec::Buffer(buffer_spec(meta)),
            HeldMeta::Geometry(meta) => data_info::Spec::Geometry(GeometrySpec {
                name: meta.name.clone(),
                vertices: meta.vertices,
                triangles: meta.triangles,
                normals: meta.normals,
                colours: meta.colours,
            }),
        }),
    }
}

pub(crate) fn proto_dtype(dtype: Dtype) -> ProtoDtype {
    match dtype {
        Dtype::Uint8 => ProtoDtype::Uint8,
        Dtype::Int8 => ProtoDtype::Int8,
        Dtype::Uint16 => ProtoDtype::Uint16,
        Dtype::Int16 => ProtoDtype::Int16,
        Dtype::Uint32 => ProtoDtype::Uint32,
        Dtype::Int32 => ProtoDtype::Int32,
        Dtype::Uint64 => ProtoDtype::Uint64,
        Dtype::Int64 => ProtoDtype::Int64,
        Dtype::Float32 => ProtoDtype::Float32,
        Dtype::Float64 => ProtoDtype::Float64,
        Dtype::Str => ProtoDtype::String,
    }
}

pub(crate) fn buffer_spec(meta: &BufferMeta) -> BufferSpec {
    BufferSpec {
        name: meta.name.clone(),
        dtype: proto_dtype(meta.dtype) as i32,
        shape: meta.shape.clone(),
        byte_length: meta.byte_length().unwrap_or_default(),
        // Left empty on the way out: a description of an array is not the array,
        // empty. A description of an array is not the array: echoing the text
        // back would put every residue name on the wire again on every
        // ListData, and the client is the side that sent them. `shape` says how
        // many there are, which is what a listing is for.
        values: Vec::new(),
    }
}
