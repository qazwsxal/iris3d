//! `SceneService`: chunked ingest and object lifecycle.
//!
//! Uploads are assembled here, on the tokio side, and only handed to the ECS
//! once complete and validated. A rejected stream never reaches the scene.

use tokio::sync::{broadcast, oneshot};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use iris3d_scene::SceneCommand;

use super::convert::{
    actor_info, data_info, filter_info, filter_kind_info, kind_info, object_info,
    params_from_proto, scene_error,
};
use super::upload::Upload;

use super::proto::{
    AddActorRequest, AddActorResponse, AddFilterRequest, AddFilterResponse, CreateObjectRequest,
    CreateObjectResponse, DataHandle, DeleteObjectRequest, DeleteObjectResponse,
    ListActorKindsRequest, ListActorKindsResponse, ListActorsRequest, ListActorsResponse,
    ListDataRequest, ListDataResponse, ListFilterKindsRequest, ListFilterKindsResponse,
    ListFiltersRequest, ListFiltersResponse, ListObjectsRequest, ListObjectsResponse, ObjectHandle,
    ReleaseDataRequest, ReleaseDataResponse, RemoveActorRequest, RemoveActorResponse,
    RemoveFilterRequest, RemoveFilterResponse, SetActorRequest, SetActorResponse, SetFilterRequest,
    SetFilterResponse, SetParentRequest, SetParentResponse, SetTransformRequest,
    SetTransformResponse, Subscribe, UploadDataRequest, UploadDataResponse, WatchRequest,
    WatchResponse, scene_service_server::SceneService, upload_data_request::Payload as DataPayload,
};
use bevy::math::{Quat, Vec3};
use bevy::prelude::warn;
use iris3d_core::bus::BusSender;
use iris3d_filter::FilterCommand;

/// Adapts the `SceneService` wire contract onto the two command channels.
///
/// Two senders rather than one, because filters are not part of the scene tree
/// and are applied by a different system over different data. Which bus a method
/// submits to is decided here, by the method, which is the only place that knows
/// both.
pub struct SceneBridgeService {
    scene: BusSender<SceneCommand>,
    filters: BusSender<FilterCommand>,
    /// Where events come from, for `Watch`. Cloning it is how each stream gets
    /// its own receiver.
    events: super::watch::Events,
}

impl SceneBridgeService {
    pub fn new(
        scene: BusSender<SceneCommand>,
        filters: BusSender<FilterCommand>,
        events: super::watch::Events,
    ) -> Self {
        Self {
            scene,
            filters,
            events,
        }
    }

    /// Submits a scene command and waits for it to be applied on the next tick.
    async fn submit<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> SceneCommand,
    ) -> Result<T, Status> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.scene
            .send(make(reply_tx))
            .map_err(|_| Status::unavailable("scene is not running"))?;
        reply_rx
            .await
            .map_err(|_| Status::internal("scene dropped the request without replying"))
    }

    /// As [`submit`](Self::submit), onto the filter bus.
    async fn submit_filter<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> FilterCommand,
    ) -> Result<T, Status> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.filters
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
        // Required. There is nothing for an empty kind to mean: the server
        // has no basis for choosing a representation, so a client asks
        // ListActorKinds and names one.
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
            .submit_filter(|reply| FilterCommand::Add {
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
            .submit_filter(|reply| FilterCommand::Set { id, params, reply })
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
            .submit_filter(|reply| FilterCommand::Remove { id, reply })
            .await?;

        Ok(Response::new(RemoveFilterResponse { removed }))
    }

    async fn list_filters(
        &self,
        _request: Request<ListFiltersRequest>,
    ) -> Result<Response<ListFiltersResponse>, Status> {
        let listing = self
            .submit_filter(|reply| FilterCommand::List { reply })
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
            .submit_filter(|reply| FilterCommand::ListKinds { reply })
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
    let kind = wanted.kinds.contains(&(event.kind as i32));
    let object = wanted.objects.is_empty()
        || wanted
            .objects
            .iter()
            .any(|handle| handle.id == event.object);
    kind && object
}
