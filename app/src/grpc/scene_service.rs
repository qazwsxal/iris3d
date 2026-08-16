//! `SceneService`: chunked ingest and object lifecycle.
//!
//! Uploads are assembled here, on the tokio side, and only handed to the ECS
//! once complete and validated. A rejected stream never reaches the scene.

use tokio::sync::{broadcast, oneshot};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::scene::registry::{ParamKind, ParamMap, ParamSpec, ParamValue};
use crate::filter::{FilterKindSummary, FilterSummary, OutputKind};
use crate::scene::{
    ActorSummary, BufferMeta, DataSummary, Dtype, HeldMeta, KindSummary, NamedBuffer,
    ObjectSummary, SceneCommand, SceneError,
};

use super::SceneSender;
use super::proto::{
    ActorHandle, ActorInfo, ActorKindInfo, AddActorRequest, AddActorResponse, AddFilterRequest,
    AddFilterResponse, ArrayOutput, ArrayParam, BoolParam, BufferSpec, ChoiceParam, Chunk,
    CreateObjectRequest, CreateObjectResponse, DataHandle, DataInfo, DeleteObjectRequest,
    DeleteObjectResponse, Dtype as ProtoDtype, FilterHandle, FilterInfo, FilterKindInfo,
    FilterOutput, FloatParam, GeometryOutput, GeometryParam, GeometrySpec, ListActorKindsRequest,
    ListActorKindsResponse, ListActorsRequest,
    ListActorsResponse, ListDataRequest, ListDataResponse, ListFilterKindsRequest,
    ListFilterKindsResponse, ListFiltersRequest, ListFiltersResponse, ListObjectsRequest,
    ListObjectsResponse, ObjectHandle, ObjectInfo, OutputSpec as ProtoOutputSpec,
    ParamSpec as ProtoSpec, ParamValue as ProtoParam, ReleaseDataRequest,
    ReleaseDataResponse, RemoveActorRequest, RemoveActorResponse, RemoveFilterRequest,
    RemoveFilterResponse, SetActorRequest, SetActorResponse, SetFilterRequest, SetFilterResponse,
    SetParentRequest, SetParentResponse, SetTransformRequest, SetTransformResponse,
    Subscribe, TextParam, UploadDataRequest, UploadDataResponse, VectorParam, WatchRequest,
    WatchResponse,
    VectorValue, data_info, output_spec, param_spec, param_value::Value,
    scene_service_server::SceneService,
    upload_data_request::Payload as DataPayload,
};
use bevy::math::{Quat, Vec3};
use bevy::prelude::warn;

/// Ceiling on the total declared size of a single object. Generous enough for
/// a large point cloud, small enough that a malformed or malicious header
/// cannot ask the process to reserve everything.
const MAX_OBJECT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Ceiling on how many named arrays one object may declare.
const MAX_BUFFERS_PER_OBJECT: usize = 64;

/// Ceiling on the text one string array may carry.
///
/// Strings travel inline in the header, which is a single gRPC message, so a
/// string array cannot grow past what the transport will decode — and hitting
/// that limit gives an opaque transport failure rather than anything a caller
/// can act on. This rejects it first, with a message that says what to do.
///
/// Well under the 8 MiB decode limit on purpose: a header carries every array's
/// declaration at once, so one array must not be able to fill it alone. A
/// million distinct names still fit, which no real name column approaches —
/// text columns are low-cardinality, and one that is not should arrive as an
/// index array and a dictionary, exactly as the hierarchy arrays do.
const MAX_TEXT_BYTES: u64 = 1024 * 1024;

/// How much of a declared buffer to reserve up front. Beyond this the vector
/// grows as bytes actually arrive, so an inflated `byte_length` costs the
/// server nothing until the client backs it up with real data.
const MAX_EAGER_RESERVE: u64 = 64 * 1024 * 1024;

/// Adapts the `SceneService` wire contract onto the scene command channel.
pub struct SceneBridgeService {
    commands: SceneSender,
    /// Where events come from, for `Watch`. Cloning it is how each stream gets
    /// its own receiver.
    events: super::watch::Events,
}

impl SceneBridgeService {
    pub fn new(commands: SceneSender, events: super::watch::Events) -> Self {
        Self { commands, events }
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

        let removed = self
            .submit(|reply| SceneCommand::DeleteObject { id, reply })
            .await?;

        Ok(Response::new(DeleteObjectResponse {
            deleted: !removed.objects.is_empty(),
            removed: removed
                .objects
                .into_iter()
                .map(|id| ObjectHandle { id })
                .collect(),
        }))
    }

    async fn add_actor(
        &self,
        request: Request<AddActorRequest>,
    ) -> Result<Response<AddActorResponse>, Status> {
        let request = request.into_inner();
        // Required. An empty kind used to mean "whatever you would have
        // chosen", and there is no longer anything to choose — ask
        // ListActorKinds and name one.
        if request.kind.is_empty() {
            return Err(Status::invalid_argument(
                "kind is required; ask ListActorKinds for the ones this build supports",
            ));
        }
        let kind = request.kind;
        // Optional, unlike the kind: with none given one is made, because
        // there is a sensible object to create and no sensible way to draw.
        // Several draws one actor in several places.
        let parents = request
            .parents
            .into_iter()
            .map(|handle| handle.id)
            .collect();
        let params = params_from_proto(request.params)?;

        let summary = self
            .submit(|reply| SceneCommand::AddActor {
                kind,
                parents,
                params,
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
        let visible = request.visible;
        // Absent leaves the placements alone; present replaces them, and an
        // empty list takes the actor off screen without removing it.
        let parents = request
            .parents
            .map(|list| list.handles.into_iter().map(|handle| handle.id).collect());

        let summary = self
            .submit(|reply| SceneCommand::SetActor {
                id,
                params,
                visible,
                parents,
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
        let parent = request.into_inner().parent.map(|handle| handle.id);

        let listing = self
            .submit(|reply| SceneCommand::ListActors { parent, reply })
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

    async fn add_filter(
        &self,
        request: Request<AddFilterRequest>,
    ) -> Result<Response<AddFilterResponse>, Status> {
        let request = request.into_inner();
        if request.kind.is_empty() {
            return Err(Status::invalid_argument(
                "kind is required; ask ListFilterKinds for the ones this build has",
            ));
        }
        let params = params_from_proto(request.params)?;

        let summary = self
            .submit(|reply| SceneCommand::AddFilter {
                kind: request.kind,
                params,
                reply,
            })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(AddFilterResponse {
            filter: Some(filter_info(&summary)),
        }))
    }

    async fn set_filter(
        &self,
        request: Request<SetFilterRequest>,
    ) -> Result<Response<SetFilterResponse>, Status> {
        let request = request.into_inner();
        let id = request
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;
        let params = params_from_proto(request.params)?;

        let summary = self
            .submit(|reply| SceneCommand::SetFilter { id, params, reply })
            .await?
            .map_err(scene_error)?;

        Ok(Response::new(SetFilterResponse {
            filter: Some(filter_info(&summary)),
        }))
    }

    async fn remove_filter(
        &self,
        request: Request<RemoveFilterRequest>,
    ) -> Result<Response<RemoveFilterResponse>, Status> {
        let id = request
            .into_inner()
            .handle
            .ok_or_else(|| Status::invalid_argument("handle is required"))?
            .id;

        let removed = self
            .submit(|reply| SceneCommand::RemoveFilter { id, reply })
            .await?;

        Ok(Response::new(RemoveFilterResponse { removed }))
    }

    async fn list_filters(
        &self,
        _request: Request<ListFiltersRequest>,
    ) -> Result<Response<ListFiltersResponse>, Status> {
        let listing = self
            .submit(|reply| SceneCommand::ListFilters { reply })
            .await?;

        Ok(Response::new(ListFiltersResponse {
            filters: listing.iter().map(filter_info).collect(),
        }))
    }

    async fn list_filter_kinds(
        &self,
        _request: Request<ListFilterKindsRequest>,
    ) -> Result<Response<ListFilterKindsResponse>, Status> {
        let kinds = self
            .submit(|reply| SceneCommand::ListFilterKinds { reply })
            .await?;

        Ok(Response::new(ListFilterKindsResponse {
            kinds: kinds
                .iter()
                .map(|kind| (kind.id.clone(), filter_kind_info(kind)))
                .collect(),
        }))
    }

    type WatchStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<WatchResponse, Status>> + Send + 'static>,
    >;

    /// Reports what the user does, and takes changes of mind while doing it.
    ///
    /// Two loops in one task: one draining the client's requests to update the
    /// subscription, one draining events to send. `tokio::select!` runs them
    /// against each other so a subscription change lands between events rather
    /// than behind however many are queued.
    ///
    /// The subscription lives *here*, per stream, rather than in the scene. The
    /// ECS reports what happened and knows nothing about who is listening —
    /// which is the same separation the command channel has in the other
    /// direction, and what keeps a registry of live gRPC clients out of the
    /// scene.
    async fn watch(
        &self,
        request: Request<Streaming<WatchRequest>>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let mut requests = request.into_inner();
        let mut events = self.events.subscribe();

        let stream = async_stream::stream! {
            // Nothing until asked. A client that opens the stream and says
            // nothing gets nothing, so the cheapest client is not the most
            // expensive to serve.
            let mut wanted: Option<Subscribe> = None;

            loop {
                tokio::select! {
                    incoming = requests.next() => match incoming {
                        Some(Ok(message)) => wanted = message.subscribe,
                        // The client hung up, or sent something unreadable.
                        // Either way there is nobody to report to.
                        Some(Err(err)) => {
                            yield Err(err);
                            break;
                        }
                        None => break,
                    },
                    event = events.recv() => match event {
                        Ok(event) => {
                            if let Some(wanted) = &wanted
                                && reportable(wanted, &event)
                            {
                                yield Ok(WatchResponse { event: Some(event.to_proto()) });
                            }
                        }
                        // Fell behind. Said out loud rather than pretended
                        // away: a client that missed clicks should know it
                        // missed them, and the alternative — an unbounded
                        // queue — lets one slow reader hold the app's memory.
                        Err(broadcast::error::RecvError::Lagged(missed)) => {
                            warn!("grpc: a watcher missed {missed} events");
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    },
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}

fn scene_error(error: SceneError) -> Status {
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

fn actor_info(summary: &ActorSummary) -> ActorInfo {
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
fn spec_to_proto(spec: &ParamSpec) -> ProtoSpec {
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

fn kind_info(summary: &KindSummary) -> ActorKindInfo {
    ActorKindInfo {
        id: summary.id.clone(),
        label: summary.label.clone(),
        params: summary.params.iter().map(spec_to_proto).collect(),
    }
}

fn filter_info(summary: &FilterSummary) -> FilterInfo {
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

fn filter_kind_info(summary: &FilterKindSummary) -> FilterKindInfo {
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

/// An upload in progress: validated metadata plus the bytes received so far.
#[derive(Debug)]
struct Upload {
    metas: Vec<BufferMeta>,
    /// Byte length each buffer must reach before the upload is complete.
    declared: Vec<u64>,
    data: Vec<Vec<u8>>,
    /// Strings taken straight from the header, one entry per declared buffer
    /// and empty for every numeric one.
    ///
    /// Complete the moment the header arrives, unlike `data`, because a string
    /// array is not chunked. Its declared length is 0, so it needs no waiting
    /// and the completeness check in `finish` passes it without a special case.
    strings: Vec<Vec<String>>,
}

impl Upload {
    /// Validates a bare list of arrays: the same checks, with no object around
    /// Validated once here so every array that reaches the scene is well formed.
    /// what counts as a well-formed declaration.
    fn open(mut specs: Vec<BufferSpec>) -> Result<Self, Status> {
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
        let mut strings = Vec::with_capacity(specs.len());
        let mut total: u64 = 0;

        for (index, spec) in specs.iter_mut().enumerate() {
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
                .checked_add(held_bytes(expected, spec))
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
            // Taken rather than cloned: the strings are the largest thing in
            // the header and the spec is not read again after validation.
            strings.push(std::mem::take(&mut spec.values));
        }

        Ok(Self {
            metas,
            declared,
            data,
            strings,
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
            .zip(self.strings)
            .map(|((meta, data), strings)| NamedBuffer {
                meta,
                data,
                strings,
            })
            .collect();

        Ok(buffers)
    }
}

/// What one declared buffer will actually occupy, against the size ceiling.
///
/// A string array declares 0 bytes and would otherwise slip past the ceiling
/// entirely, however much text it carries. The bound exists to stop one upload
/// asking for all the memory, and text is memory, so it is counted here
/// alongside the chunked bytes.
fn held_bytes(declared: u64, spec: &BufferSpec) -> u64 {
    let text: u64 = spec.values.iter().map(|value| value.len() as u64).sum();
    declared.saturating_add(text)
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

    // An array is bytes or it is text, never both: `values` belongs to
    // DTYPE_STRING alone. Accepting it beside a numeric buffer would leave two
    // sources for the same element and no rule for which wins.
    //
    // The declared `byte_length` needs no check of its own here. A string
    // array's element size is 0, so the length the caller must declare comes
    // out at 0 and the equality test every buffer already faces rejects
    // anything else.
    if dtype == Dtype::Str {
        let expected = element_count(&spec.shape);
        if spec.values.len() as u64 != expected {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): shape needs {expected} strings but values has {}",
                spec.name,
                spec.values.len()
            )));
        }

        let text: u64 = spec.values.iter().map(|value| value.len() as u64).sum();
        if text > MAX_TEXT_BYTES {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): {text} bytes of text, limit is {MAX_TEXT_BYTES}. \
                 Strings travel whole in the header, so a text array cannot grow with the \
                 element count. Send an integer index per element and a string array of the \
                 distinct values instead.",
                spec.name
            )));
        }
    } else if !spec.values.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): values is set on a {dtype} array; it belongs to string \
             arrays alone",
            spec.name
        )));
    }

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
        Ok(ProtoDtype::String) => Dtype::Str,
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

fn data_info(held: &DataSummary) -> DataInfo {
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
        Dtype::Str => ProtoDtype::String,
    }
}

fn buffer_spec(meta: &BufferMeta) -> BufferSpec {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed string array of `count` elements.
    fn text(count: usize) -> BufferSpec {
        BufferSpec {
            name: "res_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![count as u64],
            byte_length: 0,
            values: (0..count).map(|i| format!("R{i}")).collect(),
        }
    }

    /// Strings arrive complete in the header, so an upload of nothing else
    /// finishes without a single chunk. This is the case a client uploading a
    /// residue table hits, and it must not be mistaken for a truncated stream.
    #[test]
    fn a_string_array_is_complete_without_chunks() {
        let upload = Upload::open(vec![text(3)]).expect("header rejected");
        let buffers = upload.finish().expect("stream reported as truncated");

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].meta.dtype, Dtype::Str);
        assert!(buffers[0].data.is_empty());
        assert_eq!(buffers[0].strings, vec!["R0", "R1", "R2"]);
    }

    /// The shape is the count. A mismatch means the client's per-element index
    /// array and its side array disagree about how many groups there are, which
    /// silently mislabels everything if it gets through.
    #[test]
    fn the_string_count_must_match_the_shape() {
        let mut spec = text(3);
        spec.shape = vec![4];
        let error = Upload::open(vec![spec]).expect_err("count mismatch accepted");
        assert!(
            error.message().contains("needs 4 strings"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// An array is bytes or it is text. Both at once leaves two sources for the
    /// same element and no rule for which one wins.
    #[test]
    fn a_numeric_array_may_not_carry_strings() {
        let spec = BufferSpec {
            name: "positions".into(),
            dtype: ProtoDtype::Float32 as i32,
            shape: vec![1, 3],
            byte_length: 12,
            values: vec!["nope".into()],
        };
        let error = Upload::open(vec![spec]).expect_err("values accepted on a float array");
        assert!(
            error.message().contains("string arrays alone"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// A string array declares no bytes. Claiming otherwise is caught by the
    /// same equality check every buffer faces, with no special case for text.
    #[test]
    fn a_string_array_may_not_declare_bytes() {
        let mut spec = text(3);
        spec.byte_length = 12;
        let error = Upload::open(vec![spec]).expect_err("byte_length accepted on a string array");
        assert!(
            error.message().contains("byte_length is 12"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// Text is memory even though it declares no bytes, so it has to count
    /// against the same ceiling. Without this a header of nothing but strings
    /// passes every size check while asking for unbounded allocation.
    #[test]
    fn text_counts_against_the_size_ceiling() {
        // Three names of three characters each, declaring no bytes.
        assert_eq!(held_bytes(0, &text_of(&["ALA", "GLY", "HOH"])), 9);

        // A numeric buffer carries no text, so nothing changes for it.
        let numeric = BufferSpec {
            name: "positions".into(),
            dtype: ProtoDtype::Float32 as i32,
            shape: vec![1, 3],
            byte_length: 12,
            values: Vec::new(),
        };
        assert_eq!(held_bytes(12, &numeric), 12);
    }

    /// Text that outgrows the header is refused here, where the caller learns
    /// what to do about it, rather than at the transport, where it does not.
    /// This is the per-element name column a large system would otherwise send.
    #[test]
    fn text_past_the_ceiling_is_refused_with_an_answer() {
        let count = (MAX_TEXT_BYTES as usize / 4) + 1;
        let spec = BufferSpec {
            name: "atom_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![count as u64],
            byte_length: 0,
            values: vec!["CA12".into(); count],
        };
        let error = Upload::open(vec![spec]).expect_err("oversized text accepted");
        let message = error.message();
        assert!(message.contains("atom_name"), "unnamed array: {message}");
        assert!(
            message.contains("index per element"),
            "no remedy offered: {message}"
        );
    }

    /// A dictionary is what the ceiling asks for, and it fits with room to
    /// spare: the distinct values of even a large name column are a few
    /// hundred entries, whatever the element count.
    #[test]
    fn a_dictionary_of_distinct_values_fits() {
        let distinct: Vec<String> = (0..1000).map(|i| format!("name{i}")).collect();
        let spec = BufferSpec {
            name: "atom_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![distinct.len() as u64],
            byte_length: 0,
            values: distinct,
        };
        Upload::open(vec![spec]).expect("a dictionary was rejected");
    }

    /// A string array of exactly these values.
    fn text_of(values: &[&str]) -> BufferSpec {
        BufferSpec {
            name: "res_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![values.len() as u64],
            byte_length: 0,
            values: values.iter().map(|v| (*v).to_owned()).collect(),
        }
    }

    /// Never echoed: a listing describes an array, and re-sending every residue
    /// name on each ListData would make describing cost as much as uploading.
    #[test]
    fn a_listing_reports_the_count_and_not_the_text() {
        let meta = BufferMeta {
            name: "res_name".into(),
            dtype: Dtype::Str,
            shape: vec![3],
        };
        let spec = buffer_spec(&meta);
        assert_eq!(spec.dtype, ProtoDtype::String as i32);
        assert_eq!(spec.shape, vec![3]);
        assert_eq!(spec.byte_length, 0);
        assert!(spec.values.is_empty());
    }
}

/// Whether one event matches what a stream asked for.
///
/// Per stream rather than centrally — see `watch::Events`. Both filters are
/// "empty means everything except when it means nothing": an empty `kinds`
/// reports nothing, because subscribing to no kinds is how you say you are not
/// interested yet; an empty `objects` reports every object, because naming none
/// is how you say you do not care which.
///
/// The asymmetry is deliberate and is the difference between an opt-in and a
/// restriction.
fn reportable(wanted: &Subscribe, event: &super::watch::SceneEvent) -> bool {
    let kind = wanted
        .kinds
        .iter()
        .any(|asked| *asked == event.kind as i32);
    let object =
        wanted.objects.is_empty() || wanted.objects.iter().any(|handle| handle.id == event.object);
    kind && object
}
