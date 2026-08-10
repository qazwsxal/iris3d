//! `SceneService`: chunked ingest and object lifecycle.
//!
//! Uploads are assembled here, on the tokio side, and only handed to the ECS
//! once complete and validated. A rejected stream never reaches the scene.

use tokio::sync::oneshot;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::scene::actor::ColorMap;
use crate::scene::data::Association;

use crate::scene::registry::{ParamKind, ParamMap, ParamValue};
use crate::scene::subset::SubsetRequest;
use crate::scene::{
    ActorSummary, BufferMeta, ColorBy, DataSummary, Dtype, KindSummary, NamedBuffer, ObjectSummary,
    SceneCommand, SceneError, SubsetEncoding,
};

use super::SceneSender;
use super::proto::{
    ActorHandle, ActorInfo, ActorKindInfo, AddActorRequest, AddActorResponse, ArrayParam,
    BoolParam, BufferSpec, ChoiceParam, Chunk, Color, ColorSpec, CreateObjectRequest,
    CreateObjectResponse, DataHandle, DataInfo, DeleteObjectRequest, DeleteObjectResponse,
    Dtype as ProtoDtype, FloatParam, ListActorKindsRequest, ListActorKindsResponse,
    ListActorsRequest, ListActorsResponse, ListDataRequest, ListDataResponse, ListObjectsRequest,
    ListObjectsResponse, ObjectHandle, ObjectInfo, ParamSpec as ProtoSpec,
    ParamValue as ProtoParam, Range, ReleaseDataRequest, ReleaseDataResponse, RemoveActorRequest,
    RemoveActorResponse, SetActorRequest, SetActorResponse, SetParentRequest, SetParentResponse,
    SetTransformRequest, SetTransformResponse, Subset as ProtoSubset, SubsetInfo,
    UploadDataRequest, UploadDataResponse, VectorParam, VectorValue, param_spec,
    param_value::Value, scene_service_server::SceneService, subset as subset_proto,
    upload_data_request::Payload as DataPayload,
};
use bevy::color::{Color as BevyColor, ColorToComponents, Srgba};
use bevy::math::{Quat, Vec3};

/// Ceiling on the total declared size of a single object. Generous enough for
/// a large point cloud, small enough that a malformed or malicious header
/// cannot ask the process to reserve everything.
const MAX_OBJECT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Ceiling on how many named arrays one object may declare.
const MAX_BUFFERS_PER_OBJECT: usize = 64;

/// How much of a declared buffer to reserve up front. Beyond this the vector
/// grows as bytes actually arrive, so an inflated `byte_length` costs the
/// server nothing until the client backs it up with real data.
const MAX_EAGER_RESERVE: u64 = 64 * 1024 * 1024;

/// Adapts the `SceneService` wire contract onto the scene command channel.
pub struct SceneBridgeService {
    commands: SceneSender,
}

impl SceneBridgeService {
    pub fn new(commands: SceneSender) -> Self {
        Self { commands }
    }

    /// Submits a command and waits for the scene to apply it on its next tick.
    async fn submit<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> SceneCommand,
    ) -> Result<T, Status> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands
            .send(make(reply_tx))
            .map_err(|_| Status::unavailable("scene is not running"))?;
        reply_rx
            .await
            .map_err(|_| Status::internal("scene dropped the request without replying"))
    }
}

#[tonic::async_trait]
impl SceneService for SceneBridgeService {
    /// Arrays with nothing attached: no object, no place in the tree, nothing
    /// drawn. Assembled and validated here exactly as an object's buffers are,
    /// then handed over as handles for an actor to bind.
    async fn upload_data(
        &self,
        request: Request<Streaming<UploadDataRequest>>,
    ) -> Result<Response<UploadDataResponse>, Status> {
        let mut stream = request.into_inner();
        let mut upload: Option<Upload> = None;

        while let Some(message) = stream.next().await {
            match message?.payload {
                Some(DataPayload::Header(header)) => {
                    if upload.is_some() {
                        return Err(Status::invalid_argument(
                            "received a second header; one header per stream",
                        ));
                    }
                    // No name and no grid: a grid is a property of a dataset,
                    // and this call knows nothing about datasets.
                    upload = Some(Upload::open(header.arrays)?);
                }
                Some(DataPayload::Chunk(chunk)) => match upload.as_mut() {
                    Some(upload) => upload.write(chunk)?,
                    None => {
                        return Err(Status::invalid_argument(
                            "first message on the stream must be a header",
                        ));
                    }
                },
                None => return Err(Status::invalid_argument("message carried no payload")),
            }
        }

        let upload =
            upload.ok_or_else(|| Status::invalid_argument("stream closed before the header"))?;
        let buffers = upload.finish()?;
        let total_bytes = buffers.iter().map(|b| b.data.len() as u64).sum();

        let summaries = self
            .submit(|reply| SceneCommand::UploadData {
                arrays: buffers,
                reply,
            })
            .await?;

        Ok(Response::new(UploadDataResponse {
            arrays: summaries.iter().map(data_info).collect(),
            total_bytes,
        }))
    }

    async fn list_data(
        &self,
        _request: Request<ListDataRequest>,
    ) -> Result<Response<ListDataResponse>, Status> {
        let held = self
            .submit(|reply| SceneCommand::ListData { reply })
            .await?;
        Ok(Response::new(ListDataResponse {
            arrays: held.iter().map(data_info).collect(),
        }))
    }

    async fn release_data(
        &self,
        request: Request<ReleaseDataRequest>,
    ) -> Result<Response<ReleaseDataResponse>, Status> {
        let ids = request
            .into_inner()
            .handles
            .into_iter()
            .map(|handle| handle.id)
            .collect();
        let released = self
            .submit(|reply| SceneCommand::ReleaseData { ids, reply })
            .await?;
        Ok(Response::new(ReleaseDataResponse {
            released: released.into_iter().map(|id| DataHandle { id }).collect(),
        }))
    }

    async fn list_objects(
        &self,
        _request: Request<ListObjectsRequest>,
    ) -> Result<Response<ListObjectsResponse>, Status> {
        let objects = self
            .submit(|reply| SceneCommand::ListObjects { reply })
            .await?;

        Ok(Response::new(ListObjectsResponse {
            objects: objects.iter().map(object_info).collect(),
        }))
    }

    async fn create_object(
        &self,
        request: Request<CreateObjectRequest>,
    ) -> Result<Response<CreateObjectResponse>, Status> {
        let name = request.into_inner().name;
        let summary = self
            .submit(|reply| SceneCommand::CreateObject { name, reply })
            .await?;

        Ok(Response::new(CreateObjectResponse {
            handle: Some(ObjectHandle { id: summary.id }),
        }))
    }

    async fn set_parent(
        &self,
        request: Request<SetParentRequest>,
    ) -> Result<Response<SetParentResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;
        let parent = request.parent.map(|handle| handle.id);
        let keep_world_transform = request.keep_world_transform;

        self.submit(|reply| SceneCommand::SetParent {
            id,
            parent,
            keep_world_transform,
            reply,
        })
        .await?
        .map_err(scene_error)?;

        Ok(Response::new(SetParentResponse {}))
    }

    async fn set_transform(
        &self,
        request: Request<SetTransformRequest>,
    ) -> Result<Response<SetTransformResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;

        let translation = request.translation.map(|v| Vec3::new(v.x, v.y, v.z));
        let scale = request.scale.map(|v| Vec3::new(v.x, v.y, v.z));
        let rotation = request
            .rotation
            .map(|q| {
                let quat = Quat::from_xyzw(q.x, q.y, q.z, q.w);
                if quat.is_normalized() {
                    Ok(quat)
                } else if quat.length_squared() > f32::EPSILON {
                    Ok(quat.normalize())
                } else {
                    Err(Status::invalid_argument(
                        "rotation quaternion is zero length",
                    ))
                }
            })
            .transpose()?;

        self.submit(|reply| SceneCommand::SetTransform {
            id,
            translation,
            rotation,
            scale,
            reply,
        })
        .await?
        .map_err(scene_error)?;

        Ok(Response::new(SetTransformResponse {}))
    }

    async fn delete_object(
        &self,
        request: Request<DeleteObjectRequest>,
    ) -> Result<Response<DeleteObjectResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;
        let recursive = request.recursive;

        let removed = self
            .submit(|reply| SceneCommand::DeleteObject {
                id,
                recursive,
                reply,
            })
            .await?;

        Ok(Response::new(DeleteObjectResponse {
            deleted: !removed.objects.is_empty(),
            removed: removed
                .objects
                .into_iter()
                .map(|id| ObjectHandle { id })
                .collect(),
            removed_actors: removed
                .actors
                .into_iter()
                .map(|id| ActorHandle { id })
                .collect(),
        }))
    }

    async fn add_actor(
        &self,
        request: Request<AddActorRequest>,
    ) -> Result<Response<AddActorResponse>, Status> {
        let request = request.into_inner();
        let source = request
            .source
            .ok_or_else(|| Status::invalid_argument("source is required"))?
            .id;
        // Required. An empty kind used to mean "whatever you would have
        // chosen", and there is no longer anything to choose — ask
        // ListActorKinds and name one.
        if request.kind.is_empty() {
            return Err(Status::invalid_argument(
                "kind is required; ask ListActorKinds for the ones this build supports",
            ));
        }
        let kind = request.kind;
        let parent = request.parent.map(|handle| handle.id);
        let params = params_from_proto(request.params)?;
        let colour = request.color.map(colour_from_proto).transpose()?;
        let subset = request.subset.map(subset_from_proto).transpose()?;

        let summary = self
            .submit(|reply| SceneCommand::AddActor {
                source,
                kind,
                parent,
                params,
                colour,
                subset,
                reply,
            })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(AddActorResponse {
            actor: Some(actor_info(&summary)),
        }))
    }

    async fn set_actor(
        &self,
        request: Request<SetActorRequest>,
    ) -> Result<Response<SetActorResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;
        let params = params_from_proto(request.params)?;
        let colour = request.color.map(colour_from_proto).transpose()?;
        let visible = request.visible;
        // Three states, not two: leave the selection alone, replace it, or
        // clear it back to drawing everything.
        let subset = match (request.subset, request.clear_subset) {
            (Some(_), true) => {
                return Err(Status::invalid_argument(
                    "a subset and clear_subset were both given",
                ));
            }
            (Some(subset), false) => Some(Some(subset_from_proto(subset)?)),
            (None, true) => Some(None),
            (None, false) => None,
        };

        let summary = self
            .submit(|reply| SceneCommand::SetActor {
                id,
                params,
                colour,
                visible,
                subset,
                reply,
            })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(SetActorResponse {
            actor: Some(actor_info(&summary)),
        }))
    }

    async fn remove_actor(
        &self,
        request: Request<RemoveActorRequest>,
    ) -> Result<Response<RemoveActorResponse>, Status> {
        let id = request
            .into_inner()
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;

        let removed = self
            .submit(|reply| SceneCommand::RemoveActor { id, reply })
            .await?;

        Ok(Response::new(RemoveActorResponse { removed }))
    }

    async fn list_actors(
        &self,
        request: Request<ListActorsRequest>,
    ) -> Result<Response<ListActorsResponse>, Status> {
        let source = request.into_inner().source.map(|handle| handle.id);

        let listing = self
            .submit(|reply| SceneCommand::ListActors { source, reply })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(ListActorsResponse {
            actors: listing.iter().map(actor_info).collect(),
        }))
    }

    async fn list_actor_kinds(
        &self,
        _request: Request<ListActorKindsRequest>,
    ) -> Result<Response<ListActorKindsResponse>, Status> {
        let kinds = self
            .submit(|reply| SceneCommand::ListActorKinds { reply })
            .await?;

        Ok(Response::new(ListActorKindsResponse {
            kinds: kinds
                .iter()
                .map(|kind| (kind.id.clone(), kind_info(kind)))
                .collect(),
        }))
    }
}

fn scene_error(error: SceneError) -> Status {
    match error {
        SceneError::NoSuchObject(_) | SceneError::NoSuchActor(_) | SceneError::NoSuchData(_) => {
            Status::not_found(error.to_string())
        }
        // The caller's own request is at fault, and the declaration it needed in
        // order to get it right is in ListActorKinds.
        SceneError::MissingInput { .. } | SceneError::BadBinding { .. } => {
            Status::invalid_argument(error.to_string())
        }
        // The caller named something that does not exist in this build, which
        // it could have discovered with ListActorKinds.
        SceneError::UnknownKind(_) => Status::invalid_argument(error.to_string()),
        // The request was well-formed but the scene is not in a state where it
        // can be honoured.
        SceneError::WouldCycle { .. } => Status::failed_precondition(error.to_string()),
    }
}

fn params_from_proto(
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

fn params_to_proto(params: &ParamMap) -> std::collections::HashMap<String, ProtoParam> {
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
            };
            (key.clone(), ProtoParam { value: Some(value) })
        })
        .collect()
}

/// A `ColorSpec` describes colouring completely, so anything unset takes its
/// default rather than the actor's current value.
fn colour_from_proto(spec: ColorSpec) -> Result<ColorBy, Status> {
    let map = if spec.map.is_empty() {
        ColorMap::default()
    } else {
        ColorMap::from_str(&spec.map)
            .ok_or_else(|| Status::invalid_argument(format!("no colour map \"{}\"", spec.map)))?
    };

    let range = spec
        .range
        .map(|range| {
            if range.low > range.high {
                Err(Status::invalid_argument(
                    "colour range low is above its high",
                ))
            } else {
                Ok((range.low, range.high))
            }
        })
        .transpose()?;

    Ok(ColorBy {
        map,
        range,
        flat: spec
            .flat
            .map(|c| BevyColor::srgb(c.r, c.g, c.b))
            .unwrap_or(ColorBy::default().flat),
    })
}

fn colour_to_proto(colour: &ColorBy) -> ColorSpec {
    let flat = Srgba::from(colour.flat).to_f32_array();
    ColorSpec {
        map: colour.map.as_str().to_string(),
        range: colour.range.map(|(low, high)| Range { low, high }),
        flat: Some(Color {
            r: flat[0],
            g: flat[1],
            b: flat[2],
        }),
    }
}

fn actor_info(summary: &ActorSummary) -> ActorInfo {
    ActorInfo {
        handle: Some(ActorHandle { id: summary.id }),
        kind: summary.kind.clone(),
        source: Some(ObjectHandle { id: summary.source }),
        parent: summary.parent.map(|id| ObjectHandle { id }),
        params: params_to_proto(&summary.params),
        color: Some(colour_to_proto(&summary.colour)),
        visible: summary.visible,
        subset: summary.subset.map(|subset| SubsetInfo {
            encoding: match subset.encoding {
                SubsetEncoding::Indices => subset_proto::Encoding::Indices,
                SubsetEncoding::Mask => subset_proto::Encoding::Mask,
            } as i32,
            association: match subset.association {
                Association::PerPoint => subset_proto::Association::PerPoint,
                Association::PerCell => subset_proto::Association::PerCell,
            } as i32,
            selected: subset.selected,
        }),
    }
}

/// Reads a selection off the wire.
///
/// The values stay raw here: this runs on the transport thread with no access
/// to the world, so — exactly as an upload does — the bytes cross the channel
/// and the scene turns them into a shared asset on its own tick.
fn subset_from_proto(subset: ProtoSubset) -> Result<SubsetRequest, Status> {
    let dtype = decode_dtype(subset.dtype)
        .ok_or_else(|| Status::invalid_argument("subset dtype is required"))?;
    let encoding = match subset_proto::Encoding::try_from(subset.encoding) {
        Ok(subset_proto::Encoding::Indices) => SubsetEncoding::Indices,
        Ok(subset_proto::Encoding::Mask) => SubsetEncoding::Mask,
        _ => {
            return Err(Status::invalid_argument(
                "subset encoding must be indices or mask",
            ));
        }
    };
    let association = match subset_proto::Association::try_from(subset.association) {
        Ok(subset_proto::Association::PerCell) => Association::PerCell,
        // Per-point is the common case and the sensible reading of "unset".
        _ => Association::PerPoint,
    };

    let width = dtype.size();
    if !(subset.data.len() as u64).is_multiple_of(width) {
        return Err(Status::invalid_argument(format!(
            "subset has {} bytes, which is not a whole number of {dtype} values",
            subset.data.len()
        )));
    }
    if subset.data.is_empty() {
        return Err(Status::invalid_argument("subset is empty"));
    }

    Ok(SubsetRequest {
        data: subset.data,
        dtype,
        encoding,
        association,
    })
}

fn kind_info(summary: &KindSummary) -> ActorKindInfo {
    ActorKindInfo {
        id: summary.id.clone(),
        label: summary.label.clone(),
        params: summary
            .params
            .iter()
            .map(|spec| ProtoSpec {
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
                    ParamKind::Choice { options, default } => {
                        param_spec::Kind::Choice(ChoiceParam {
                            options: options.iter().map(|option| option.to_string()).collect(),
                            default_value: default.to_string(),
                        })
                    }
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
                    ParamKind::Array {
                        dtypes,
                        shape,
                        required,
                    } => param_spec::Kind::Array(ArrayParam {
                        dtypes: dtypes
                            .iter()
                            .map(|dtype| proto_dtype(*dtype) as i32)
                            .collect(),
                        shape: shape.to_vec(),
                        required,
                    }),
                }),
            })
            .collect(),
    }
}

/// An upload in progress: validated metadata plus the bytes received so far.
struct Upload {
    metas: Vec<BufferMeta>,
    /// Byte length each buffer must reach before the upload is complete.
    declared: Vec<u64>,
    data: Vec<Vec<u8>>,
}

impl Upload {
    /// Validates a bare list of arrays: the same checks, with no object around
    /// Validated once here so every array that reaches the scene is well formed.
    /// what counts as a well-formed declaration.
    fn open(specs: Vec<BufferSpec>) -> Result<Self, Status> {
        if specs.is_empty() {
            return Err(Status::invalid_argument("header declared no buffers"));
        }
        if specs.len() > MAX_BUFFERS_PER_OBJECT {
            return Err(Status::invalid_argument(format!(
                "header declared {} buffers, limit is {MAX_BUFFERS_PER_OBJECT}",
                specs.len()
            )));
        }

        let mut metas = Vec::with_capacity(specs.len());
        let mut declared = Vec::with_capacity(specs.len());
        let mut data = Vec::with_capacity(specs.len());
        let mut total: u64 = 0;

        for (index, spec) in specs.iter().enumerate() {
            let meta = buffer_meta(index, spec)?;

            let expected = meta.byte_length().ok_or_else(|| {
                Status::invalid_argument(format!("buffer {index}: shape overflows a u64"))
            })?;
            if expected != spec.byte_length {
                return Err(Status::invalid_argument(format!(
                    "buffer {index} (\"{}\"): byte_length is {} but {} {} elements need {expected}",
                    meta.name,
                    spec.byte_length,
                    meta.dtype,
                    element_count(&meta.shape),
                )));
            }

            total = total
                .checked_add(expected)
                .ok_or_else(|| Status::invalid_argument("declared object size overflows a u64"))?;
            if total > MAX_OBJECT_BYTES {
                return Err(Status::invalid_argument(format!(
                    "object declares {total} bytes, limit is {MAX_OBJECT_BYTES}"
                )));
            }

            if metas
                .iter()
                .any(|other: &BufferMeta| other.name == meta.name)
            {
                return Err(Status::invalid_argument(format!(
                    "duplicate buffer name \"{}\"",
                    meta.name
                )));
            }

            metas.push(meta);
            declared.push(expected);
            data.push(Vec::with_capacity(expected.min(MAX_EAGER_RESERVE) as usize));
        }

        Ok(Self {
            metas,
            declared,
            data,
        })
    }

    /// Appends one chunk, rejecting anything out of order or oversized.
    fn write(&mut self, chunk: Chunk) -> Result<(), Status> {
        let index = chunk.buffer_index as usize;
        let buffer = self.data.get_mut(index).ok_or_else(|| {
            Status::invalid_argument(format!(
                "chunk names buffer {index}, header declared {}",
                self.declared.len()
            ))
        })?;

        let written = buffer.len() as u64;
        if chunk.offset != written {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: chunk offset {} does not continue from {written}",
                chunk.offset
            )));
        }
        if chunk.data.is_empty() {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: empty chunk at offset {written}"
            )));
        }

        let remaining = self.declared[index] - written;
        if chunk.data.len() as u64 > remaining {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: chunk of {} bytes overruns the declared length by {}",
                chunk.data.len(),
                chunk.data.len() as u64 - remaining
            )));
        }

        buffer.extend_from_slice(&chunk.data);
        Ok(())
    }

    /// Confirms every array is complete and yields the finished bytes.
    fn finish(self) -> Result<Vec<NamedBuffer>, Status> {
        for (index, (buffer, declared)) in self.data.iter().zip(&self.declared).enumerate() {
            if buffer.len() as u64 != *declared {
                return Err(Status::data_loss(format!(
                    "buffer {index} (\"{}\"): stream ended with {} of {declared} bytes",
                    self.metas[index].name,
                    buffer.len()
                )));
            }
        }

        let buffers = self
            .metas
            .into_iter()
            .zip(self.data)
            .map(|(meta, data)| NamedBuffer { meta, data })
            .collect();

        Ok(buffers)
    }
}

/// Validates one `BufferSpec` and converts it to its domain equivalent.
fn buffer_meta(index: usize, spec: &BufferSpec) -> Result<BufferMeta, Status> {
    if spec.name.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index}: name is required"
        )));
    }
    if spec.shape.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): shape is required",
            spec.name
        )));
    }
    if spec.shape.contains(&0) {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): shape has a zero-length axis",
            spec.name
        )));
    }

    let dtype = match decode_dtype(spec.dtype) {
        Some(dtype) => dtype,
        None => {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): unknown dtype {}",
                spec.name, spec.dtype
            )));
        }
    };

    Ok(BufferMeta {
        name: spec.name.clone(),
        dtype,
        shape: spec.shape.clone(),
    })
}

/// Wire dtype to its domain equivalent, `None` for unset or unrecognised.
fn decode_dtype(dtype: i32) -> Option<Dtype> {
    Some(match ProtoDtype::try_from(dtype) {
        Ok(ProtoDtype::Uint8) => Dtype::Uint8,
        Ok(ProtoDtype::Int8) => Dtype::Int8,
        Ok(ProtoDtype::Uint16) => Dtype::Uint16,
        Ok(ProtoDtype::Int16) => Dtype::Int16,
        Ok(ProtoDtype::Uint32) => Dtype::Uint32,
        Ok(ProtoDtype::Int32) => Dtype::Int32,
        Ok(ProtoDtype::Uint64) => Dtype::Uint64,
        Ok(ProtoDtype::Int64) => Dtype::Int64,
        Ok(ProtoDtype::Float32) => Dtype::Float32,
        Ok(ProtoDtype::Float64) => Dtype::Float64,
        Ok(ProtoDtype::Unspecified) | Err(_) => return None,
    })
}

fn element_count(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn object_info(summary: &ObjectSummary) -> ObjectInfo {
    ObjectInfo {
        handle: Some(ObjectHandle { id: summary.id }),
        name: summary.name.clone(),
        actors: summary.actors.iter().map(actor_info).collect(),
        parent: summary.parent.map(|id| ObjectHandle { id }),
    }
}

fn data_info(array: &DataSummary) -> DataInfo {
    DataInfo {
        handle: Some(DataHandle { id: array.id }),
        spec: Some(buffer_spec(&array.meta)),
    }
}

fn proto_dtype(dtype: Dtype) -> ProtoDtype {
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
    }
}

fn buffer_spec(meta: &BufferMeta) -> BufferSpec {
    BufferSpec {
        name: meta.name.clone(),
        dtype: proto_dtype(meta.dtype) as i32,
        shape: meta.shape.clone(),
        byte_length: meta.byte_length().unwrap_or_default(),
    }
}
