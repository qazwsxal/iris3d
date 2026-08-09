//! `SceneService`: chunked ingest and object lifecycle.
//!
//! Uploads are assembled here, on the tokio side, and only handed to the ECS
//! once complete and validated. A rejected stream never reaches the scene.

use crossbeam_channel::Sender;
use tokio::sync::oneshot;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::scene::registry::{ParamKind, ParamMap, ParamValue};
use crate::scene::representation::ColorMap;
use crate::scene::{
    BufferMeta, ColorBy, Dtype, KindSummary, NamedBuffer, ObjectSummary, RepresentationSummary,
    SceneCommand, SceneError,
};

use super::proto::{
    AddRepresentationRequest, AddRepresentationResponse, BoolParam, BufferSpec, Chunk, Color,
    ColorSpec, CreateObjectRequest, CreateObjectResponse, DeleteObjectRequest,
    DeleteObjectResponse, Dtype as ProtoDtype, FloatParam, ListObjectsRequest, ListObjectsResponse,
    ListRepresentationKindsRequest, ListRepresentationKindsResponse, ListRepresentationsRequest,
    ListRepresentationsResponse, ObjectHandle, ObjectHeader, ObjectInfo, ParamSpec as ProtoSpec,
    ParamValue as ProtoParam, Range, RemoveRepresentationRequest, RemoveRepresentationResponse,
    RepresentationHandle, RepresentationInfo, RepresentationKindInfo, SetParentRequest,
    SetParentResponse, SetRepresentationRequest, SetRepresentationResponse, SetTransformRequest,
    SetTransformResponse, UploadObjectRequest, UploadObjectResponse, param_spec,
    param_value::Value, scene_service_server::SceneService, upload_object_request::Payload,
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
    commands: Sender<SceneCommand>,
}

impl SceneBridgeService {
    pub fn new(commands: Sender<SceneCommand>) -> Self {
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
    async fn upload_object(
        &self,
        request: Request<Streaming<UploadObjectRequest>>,
    ) -> Result<Response<UploadObjectResponse>, Status> {
        let mut stream = request.into_inner();
        let mut upload: Option<Upload> = None;

        while let Some(message) = stream.next().await {
            match message?.payload {
                Some(Payload::Header(header)) => {
                    if upload.is_some() {
                        return Err(Status::invalid_argument(
                            "received a second header; one object per stream",
                        ));
                    }
                    upload = Some(Upload::open(header)?);
                }
                Some(Payload::Chunk(chunk)) => match upload.as_mut() {
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
        let (name, buffers) = upload.finish()?;
        let total_bytes = buffers.iter().map(|b| b.data.len() as u64).sum();

        let summary = self
            .submit(|reply| SceneCommand::InsertObject {
                name,
                buffers,
                reply,
            })
            .await?;

        Ok(Response::new(UploadObjectResponse {
            handle: Some(ObjectHandle { id: summary.id }),
            total_bytes,
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
                    Err(Status::invalid_argument("rotation quaternion is zero length"))
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
            removed_representations: removed
                .representations
                .into_iter()
                .map(|id| RepresentationHandle { id })
                .collect(),
        }))
    }

    async fn add_representation(
        &self,
        request: Request<AddRepresentationRequest>,
    ) -> Result<Response<AddRepresentationResponse>, Status> {
        let request = request.into_inner();
        let source = request
            .source
            .ok_or_else(|| Status::invalid_argument("source is required"))?
            .id;
        // Empty rather than absent means "whatever you would have chosen", so
        // there is no separate way to say it.
        let kind = (!request.kind.is_empty()).then_some(request.kind);
        let parent = request.parent.map(|handle| handle.id);
        let params = params_from_proto(request.params)?;
        let colour = request.color.map(colour_from_proto).transpose()?;

        let summary = self
            .submit(|reply| SceneCommand::AddRepresentation {
                source,
                kind,
                parent,
                params,
                colour,
                reply,
            })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(AddRepresentationResponse {
            representation: Some(representation_info(&summary)),
        }))
    }

    async fn set_representation(
        &self,
        request: Request<SetRepresentationRequest>,
    ) -> Result<Response<SetRepresentationResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;
        let params = params_from_proto(request.params)?;
        let colour = request.color.map(colour_from_proto).transpose()?;
        let visible = request.visible;

        let summary = self
            .submit(|reply| SceneCommand::SetRepresentation {
                id,
                params,
                colour,
                visible,
                reply,
            })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(SetRepresentationResponse {
            representation: Some(representation_info(&summary)),
        }))
    }

    async fn remove_representation(
        &self,
        request: Request<RemoveRepresentationRequest>,
    ) -> Result<Response<RemoveRepresentationResponse>, Status> {
        let id = request
            .into_inner()
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;

        let removed = self
            .submit(|reply| SceneCommand::RemoveRepresentation { id, reply })
            .await?;

        Ok(Response::new(RemoveRepresentationResponse { removed }))
    }

    async fn list_representations(
        &self,
        request: Request<ListRepresentationsRequest>,
    ) -> Result<Response<ListRepresentationsResponse>, Status> {
        let source = request.into_inner().source.map(|handle| handle.id);

        let listing = self
            .submit(|reply| SceneCommand::ListRepresentations { source, reply })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(ListRepresentationsResponse {
            representations: listing.iter().map(representation_info).collect(),
        }))
    }

    async fn list_representation_kinds(
        &self,
        _request: Request<ListRepresentationKindsRequest>,
    ) -> Result<Response<ListRepresentationKindsResponse>, Status> {
        let kinds = self
            .submit(|reply| SceneCommand::ListRepresentationKinds { reply })
            .await?;

        Ok(Response::new(ListRepresentationKindsResponse {
            kinds: kinds.iter().map(kind_info).collect(),
        }))
    }
}

fn scene_error(error: SceneError) -> Status {
    match error {
        SceneError::NoSuchObject(_) | SceneError::NoSuchRepresentation(_) => {
            Status::not_found(error.to_string())
        }
        // The caller named something that does not exist in this build, which
        // it could have discovered with ListRepresentationKinds.
        SceneError::UnknownKind(_) => Status::invalid_argument(error.to_string()),
        // The request was well-formed but the scene is not in a state where it
        // can be honoured.
        SceneError::WouldCycle { .. } | SceneError::KindNotSupported { .. } => {
            Status::failed_precondition(error.to_string())
        }
    }
}

fn params_from_proto(params: std::collections::HashMap<String, ProtoParam>) -> Result<ParamMap, Status> {
    params
        .into_iter()
        .map(|(key, value)| {
            let value = match value.value {
                Some(Value::Number(number)) => ParamValue::Float(number as f32),
                Some(Value::Flag(flag)) => ParamValue::Bool(flag),
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
            };
            (
                key.clone(),
                ProtoParam {
                    value: Some(value),
                },
            )
        })
        .collect()
}

/// A `ColorSpec` describes colouring completely, so anything unset takes its
/// default rather than the representation's current value.
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
        field: spec.field,
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
        field: colour.field.clone(),
        map: colour.map.as_str().to_string(),
        range: colour.range.map(|(low, high)| Range { low, high }),
        flat: Some(Color {
            r: flat[0],
            g: flat[1],
            b: flat[2],
        }),
    }
}

fn representation_info(summary: &RepresentationSummary) -> RepresentationInfo {
    RepresentationInfo {
        handle: Some(RepresentationHandle { id: summary.id }),
        kind: summary.kind.clone(),
        source: Some(ObjectHandle { id: summary.source }),
        parent: summary.parent.map(|id| ObjectHandle { id }),
        params: params_to_proto(&summary.params),
        color: Some(colour_to_proto(&summary.colour)),
        visible: summary.visible,
    }
}

fn kind_info(summary: &KindSummary) -> RepresentationKindInfo {
    RepresentationKindInfo {
        id: summary.id.clone(),
        label: summary.label.clone(),
        supports: summary.supports.clone(),
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
                }),
            })
            .collect(),
    }
}

/// An upload in progress: validated metadata plus the bytes received so far.
struct Upload {
    name: String,
    metas: Vec<BufferMeta>,
    /// Byte length each buffer must reach before the upload is complete.
    declared: Vec<u64>,
    data: Vec<Vec<u8>>,
}

impl Upload {
    /// Validates a header and allocates the buffers it declares.
    fn open(header: ObjectHeader) -> Result<Self, Status> {
        if header.buffers.is_empty() {
            return Err(Status::invalid_argument("header declared no buffers"));
        }
        if header.buffers.len() > MAX_BUFFERS_PER_OBJECT {
            return Err(Status::invalid_argument(format!(
                "header declared {} buffers, limit is {MAX_BUFFERS_PER_OBJECT}",
                header.buffers.len()
            )));
        }

        let mut metas = Vec::with_capacity(header.buffers.len());
        let mut declared = Vec::with_capacity(header.buffers.len());
        let mut data = Vec::with_capacity(header.buffers.len());
        let mut total: u64 = 0;

        for (index, spec) in header.buffers.iter().enumerate() {
            let meta = buffer_meta(index, spec)?;

            let expected = meta.byte_length().ok_or_else(|| {
                Status::invalid_argument(format!("buffer {index}: shape overflows a u64"))
            })?;
            if expected != spec.byte_length {
                return Err(Status::invalid_argument(format!(
                    "buffer {index} (\"{}\"): byte_length is {} but {} {} elements need {expected}",
                    meta.name, spec.byte_length, meta.dtype, element_count(&meta.shape),
                )));
            }

            total = total.checked_add(expected).ok_or_else(|| {
                Status::invalid_argument("declared object size overflows a u64")
            })?;
            if total > MAX_OBJECT_BYTES {
                return Err(Status::invalid_argument(format!(
                    "object declares {total} bytes, limit is {MAX_OBJECT_BYTES}"
                )));
            }

            if metas.iter().any(|other: &BufferMeta| other.name == meta.name) {
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
            name: header.name,
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

    /// Confirms every buffer is complete and yields the finished object.
    fn finish(self) -> Result<(String, Vec<NamedBuffer>), Status> {
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

        Ok((self.name, buffers))
    }
}

/// Validates one `BufferSpec` and converts it to its domain equivalent.
fn buffer_meta(index: usize, spec: &BufferSpec) -> Result<BufferMeta, Status> {
    if spec.name.is_empty() {
        return Err(Status::invalid_argument(format!("buffer {index}: name is required")));
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

    let dtype = match ProtoDtype::try_from(spec.dtype) {
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
        Ok(ProtoDtype::Unspecified) | Err(_) => {
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

fn element_count(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn object_info(summary: &ObjectSummary) -> ObjectInfo {
    ObjectInfo {
        handle: Some(ObjectHandle { id: summary.id }),
        name: summary.name.clone(),
        buffers: summary.buffers.iter().map(buffer_spec).collect(),
        total_bytes: summary.total_bytes,
        dataset_kind: summary.kind.as_str().to_string(),
        drawn_by: summary.representations.iter().map(representation_info).collect(),
        parent: summary.parent.map(|id| ObjectHandle { id }),
    }
}

fn buffer_spec(meta: &BufferMeta) -> BufferSpec {
    let dtype = match meta.dtype {
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
    };

    BufferSpec {
        name: meta.name.clone(),
        dtype: dtype as i32,
        shape: meta.shape.clone(),
        byte_length: meta.byte_length().unwrap_or_default(),
    }
}
