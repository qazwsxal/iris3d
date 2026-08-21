//! The data commands: taking arrays in, listing them, and forgetting them.
//!
//! Arrays belong to no object, so these three are the whole of what the scene
//! does with data. Everything else names a handle and lets the store answer.

use bevy::prelude::*;

use crate::counter::GlobalIDCounter;
use crate::data::{DataArray, DataStore, HeldMeta};
use crate::model::SceneError;

use super::DataSummary;

/// Takes arrays in with no object around them. The bytes become assets exactly
/// as an object's buffers would; the only difference is who holds the handle.
pub(crate) fn upload_data(
    counter: &mut GlobalIDCounter,
    arrays: &mut Assets<DataArray>,
    store: &mut DataStore,
    uploaded: Vec<crate::data::NamedBuffer>,
) -> Vec<DataSummary> {
    let summaries: Vec<(DataSummary, String)> = uploaded
        .into_iter()
        .map(|buffer| {
            let id = counter.next_id();
            let meta = buffer.meta;
            let handle = arrays.add(DataArray {
                dtype: meta.dtype,
                shape: meta.shape.clone(),
                data: buffer.data,
                strings: buffer.strings,
            });
            store.insert(id, meta.clone(), handle);
            let name = meta.name.clone();
            (
                DataSummary {
                    id,
                    meta: HeldMeta::Array(meta),
                },
                name,
            )
        })
        .collect::<Vec<_>>();
    info!(
        "scene: took in {} array(s): {}",
        summaries.len(),
        summaries
            .iter()
            .map(|(array, name)| format!("{}={name}", array.id))
            .collect::<Vec<_>>()
            .join(" ")
    );
    summaries.into_iter().map(|(summary, _)| summary).collect()
}

/// Arrays first, then meshes, each in handle order.
///
/// Both are things a client holds and can bind, so both belong in one listing
/// rather than making geometry a call of its own.
pub(crate) fn list_data(store: &DataStore) -> Vec<DataSummary> {
    store
        .iter()
        .map(|(id, array)| DataSummary {
            id,
            meta: HeldMeta::Array(array.meta.clone()),
        })
        .chain(store.iter_geometry().map(|(id, mesh)| DataSummary {
            id,
            meta: HeldMeta::Geometry(mesh.meta.clone()),
        }))
        .collect()
}

/// Forgets arrays, reporting which of them were held.
pub(crate) fn release_data(store: &mut DataStore, ids: Vec<u64>) -> Vec<u64> {
    ids.into_iter()
        .filter(|id| {
            // An array a filter writes is not the caller's to forget: releasing
            // it would leave the filter producing into nothing, which looks
            // exactly like a filter that has broken. Reported rather than
            // refused, because the call takes a batch and one bad handle should
            // not lose the others.
            match store.generated_by(*id) {
                Some(filter) => {
                    warn!("{}", SceneError::StillGenerated { data: *id, filter });
                    false
                }
                None => store.remove(*id),
            }
        })
        .collect()
}
