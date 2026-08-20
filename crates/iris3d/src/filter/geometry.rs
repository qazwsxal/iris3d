//! Loose arrays in, one mesh out.
//!
//! The assembly step. Positions, triangles and whatever per-vertex arrays go
//! with them become a single [`Mesh`] that every consumer **references** —
//! `Mesh3d(handle)` on two entities draws the same vertex buffers twice with two
//! materials.
//!
//! # Why this is a filter and not something each actor does
//!
//! If each actor assembled its own mesh, `surface` and `medium` would both take
//! positions, indices, normals and colour and build their own `bevy::Mesh` from
//! them — so a ribbon drawn as a lit surface *and* as an absorbing medium would
//! be the same vertices uploaded twice, and a third way of drawing it a third
//! copy. Assembling once here is the other half of the same argument that makes
//! the cartoon a filter.
//!
//! Assembling once and sharing the handle also puts the question "how do these
//! arrays become a mesh" in one place instead of in every kind that draws
//! triangles. There was one copy of the index-range check, the subset remap and
//! the normal fallback per kind, and they had drifted.
//!
//! # It is also the path for an uploaded mesh
//!
//! A client that uploaded positions and indices has arrays, not geometry, so it
//! runs them through here first. That is deliberately the *same* path a filter's
//! output takes rather than a second one: `surface` has one input and no idea
//! whether what it draws was computed or uploaded.
//!
//! # Colour arrives here, not at the actor
//!
//! A consumer cannot add its own vertex colours to a mesh it shares — two actors
//! writing the same buffer would each see the other's. So the `colormap` output
//! is bound *here*, and two actors over one geometry are two views of one
//! colouring. Wanting them coloured differently means two geometry filters, and
//! that is the honest cost: different colours per vertex really are different
//! vertex buffers.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::Dtype;
use crate::scene::subset::Remap;
use iris3d_model::{ParamKind, ParamSpec};

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "positions",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    // Triangles only, and the shape says so. A tetrahedral or line connectivity
    // array cannot be bound here at all, and the caller is told why by the call
    // that tried. Drawing a volumetric mesh means extracting its boundary faces
    // first; lines want a line actor.
    ParamSpec {
        id: "indices",
        label: "triangles",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    // Unbound means "work them out from the triangles", which is what happened
    // when an upload carried none. Smooth, because the mesh is indexed.
    ParamSpec {
        id: "normals",
        label: "normals",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: false,
            structural: true,
        },
    },
    // Linear RGB, one triple per vertex, already mapped. Which ramp and what
    // range produced them is the `colormap` filter's business.
    ParamSpec {
        id: "colour",
        label: "colour",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: false,
            structural: true,
        },
    },
    // Which vertices to keep, by index. Narrowing has to happen *here*: two
    // actors sharing one mesh cannot each narrow it, and narrowing after
    // assembly would mean rewriting the connectivity of geometry somebody else
    // is drawing.
    //
    // A triangle survives only when all three of its corners do, following VTK's
    // extract-selection — keeping one with a dropped corner would mean inventing
    // a position for it.
    ParamSpec {
        id: "vertices",
        label: "vertices to keep",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
];

const OUTPUTS: &[OutputSpec] = &[
    OutputSpec {
        id: "geometry",
        label: "geometry",
        kind: OutputKind::Geometry,
        // Vertex i of the mesh is `positions[kept[i]]` — see "kept" below,
        // which is what makes a picked vertex traceable back through
        // whatever produced "positions", the same way `reindex` already
        // lets a narrowed array be traced through its own `kept`.
        provenance: Provenance::Map {
            via: "kept",
            of: "positions",
        },
    },
    OutputSpec {
        id: "kept",
        label: "kept vertices",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[0],
        },
        // The map itself; nothing walks further back through it.
        provenance: Provenance::Opaque,
    },
];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "geometry",
        label: "geometry",
        params: PARAMS,
        outputs: OUTPUTS,
        run: Some(run),
    });
}

fn run(request: &Request) -> Outcome {
    let mut products = Products::new();
    let (Some(position_array), Some(index_array)) =
        (request.input("positions"), request.input("indices"))
    else {
        return Outcome::refused("has nothing bound to \"positions\" or \"indices\"");
    };
    let all = position_array.to_vec3();
    let Some(all_indices) = index_array.to_u32() else {
        return Outcome::refused("was given triangles that are not an integer type");
    };
    if all.is_empty() || all_indices.is_empty() {
        return Outcome::refused("was given no vertices or no triangles");
    }
    if let Some(out_of_range) = all_indices.iter().find(|i| **i as usize >= all.len()) {
        return Outcome::refused(format!(
            "has triangle index {out_of_range}, past its {} vertices",
            all.len()
        ));
    }

    // Read against the *unsubsetted* count, so a per-vertex array that does not
    // match the positions is dropped rather than being silently offset.
    let per_vertex = |id: &str| -> Option<Vec<Vec3>> {
        request
            .input(id)
            .map(|array| array.to_vec3())
            .filter(|values| values.len() == all.len())
    };
    let normals = per_vertex("normals");
    let colours = per_vertex("colour");

    let kept = kept_vertices(request, all.len());
    let (positions, indices) = match &kept {
        Some(kept) => {
            let remap = Remap::new(kept, all.len());
            let indices: Vec<u32> = all_indices
                .chunks_exact(3)
                .filter_map(|corners| remap.cell(corners))
                .flatten()
                .collect();
            if indices.is_empty() {
                return Outcome::refused(
                    "kept no whole triangles: every triangle had at least one \
                     vertex cut by \"vertices to keep\"",
                );
            }
            let positions = kept.iter().map(|index| all[*index as usize]).collect();
            (positions, indices)
        }
        None => (all, all_indices),
    };
    let narrow = |values: Vec<Vec3>| -> Vec<Vec3> {
        match &kept {
            Some(kept) => kept.iter().map(|index| values[*index as usize]).collect(),
            None => values,
        }
    };

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, triples(&positions));
    mesh.insert_indices(Indices::U32(indices));

    match normals.map(narrow) {
        Some(normals) => mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, triples(&normals)),
        // Smooth, because the mesh is indexed. Faceting a surface that is meant
        // to read as continuous turns one highlight into a mosaic of them.
        None => mesh.compute_normals(),
    }

    // Opaque. The alpha of a vertex colour means nothing to either consumer —
    // a lit surface is opaque and a medium's transparency is its absorbance — so
    // there is no fourth channel to carry and nothing to read one from.
    if let Some(colours) = colours.map(narrow) {
        let rgba: Vec<[f32; 4]> = colours.iter().map(|c| [c.x, c.y, c.z, 1.0]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, rgba);
    }

    // Always produced, identity when nothing was cut, so a pick can walk
    // "geometry" back through "kept" the same way whether or not `vertices`
    // was bound.
    let kept_indices: Vec<u32> = match &kept {
        Some(kept) => kept.clone(),
        None => (0..positions.len() as u32).collect(),
    };

    debug!("geometry: assembled {} vertices", mesh.count_vertices());
    products.insert("geometry", mesh.into());
    products.insert(
        "kept",
        crate::scene::DataArray::numeric(
            Dtype::Uint32,
            vec![kept_indices.len() as u64],
            kept_indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
        )
        .into(),
    );
    products.into()
}

/// Which vertices the `vertices` input selects, or `None` for all of them.
///
/// `None` also covers a selection that names everything or nothing: the first is
/// the same as no selection and the second is more likely a mistake than a
/// request to draw an empty mesh. Both match what an actor's `Subset` did.
fn kept_vertices(request: &Request, count: usize) -> Option<Vec<u32>> {
    let selected = request.input("vertices")?.to_u32()?;
    if selected.is_empty() {
        warn!("geometry: the selection is empty; keeping everything");
        return None;
    }
    if selected.len() == count {
        return None;
    }
    Some(selected)
}

fn triples(values: &[Vec3]) -> Vec<[f32; 3]> {
    values.iter().map(|v| [v.x, v.y, v.z]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::DataArray;
    use bevy::platform::collections::HashMap;
    use iris3d_model::{ParamMap, ParamValue};

    fn floats(values: &[f32], shape: Vec<u64>) -> DataArray {
        DataArray::numeric(
            Dtype::Float32,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    fn indices(values: &[u32]) -> DataArray {
        DataArray::numeric(
            Dtype::Uint32,
            vec![values.len() as u64 / 3, 3],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// Two triangles sharing an edge: four vertices, six indices.
    fn quad() -> HashMap<&'static str, DataArray> {
        let mut inputs = HashMap::new();
        inputs.insert(
            "positions",
            floats(
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![4, 3],
            ),
        );
        inputs.insert("indices", indices(&[0, 1, 2, 0, 2, 3]));
        inputs
    }

    fn request(inputs: HashMap<&'static str, DataArray>) -> Request {
        Request {
            params: ParamMap::default(),
            inputs,
        }
    }

    fn mesh(products: &Outcome) -> &Mesh {
        products.products["geometry"]
            .geometry()
            .expect("geometry is a mesh")
    }

    #[test]
    fn assembles_one_mesh_and_derives_the_normals() {
        let products = run(&request(quad()));
        let mesh = mesh(&products);
        assert_eq!(mesh.count_vertices(), 4);
        assert_eq!(mesh.indices().map(|i| i.len()), Some(6));
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some(),
            "unbound normals should be worked out from the triangles"
        );
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_none());
    }

    /// `kept` is what lets a picked vertex be walked back to "positions". With
    /// nothing cut, vertex i of the mesh is still positions[i], so `kept` has
    /// to say so explicitly rather than being absent.
    #[test]
    fn kept_is_the_identity_when_nothing_is_cut() {
        let products = run(&request(quad()));
        let kept = products.products["kept"]
            .array()
            .expect("kept is an array")
            .to_u32()
            .expect("integers");
        assert_eq!(kept, vec![0, 1, 2, 3]);
    }

    /// A selection renumbers the mesh, so `kept` has to name the *original*
    /// index behind each surviving vertex, in the mesh's new order.
    #[test]
    fn kept_maps_a_narrowed_vertex_back_to_its_original_index() {
        let mut inputs = quad();
        inputs.insert(
            "vertices",
            DataArray::numeric(
                Dtype::Uint32,
                vec![3],
                [0u32, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
        let products = run(&request(inputs));
        let kept = products.products["kept"]
            .array()
            .expect("kept is an array")
            .to_u32()
            .expect("integers");
        assert_eq!(kept, vec![0, 1, 2]);
    }

    /// The whole reason colour is bound here rather than at the actor: two
    /// actors share the buffer, so the colouring has to be part of it.
    #[test]
    fn a_bound_colour_becomes_a_vertex_attribute() {
        let mut inputs = quad();
        inputs.insert("colour", floats(&[0.5; 12], vec![4, 3]));
        let products = run(&request(inputs));
        assert!(mesh(&products).attribute(Mesh::ATTRIBUTE_COLOR).is_some());
    }

    /// A colour array that does not match the positions is dropped rather than
    /// applied to whichever vertices it happens to reach — better an untinted
    /// mesh than one whose colours are offset from its vertices.
    #[test]
    fn a_mismatched_colour_is_ignored() {
        let mut inputs = quad();
        inputs.insert("colour", floats(&[0.5; 6], vec![2, 3]));
        let products = run(&request(inputs));
        assert!(mesh(&products).attribute(Mesh::ATTRIBUTE_COLOR).is_none());
    }

    /// Selecting vertices renumbers the triangles rather than merely filtering
    /// them, and a triangle survives only when all three corners do.
    #[test]
    fn a_selection_rewrites_the_connectivity() {
        let mut inputs = quad();
        // Drops vertex 3, which only the second triangle uses.
        inputs.insert(
            "vertices",
            DataArray::numeric(
                Dtype::Uint32,
                vec![3],
                [0u32, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
        let products = run(&request(inputs));
        let mesh = mesh(&products);
        assert_eq!(mesh.count_vertices(), 3);
        assert_eq!(
            mesh.indices().map(|i| i.len()),
            Some(3),
            "the triangle with a dropped corner should go with it"
        );
    }

    /// An index past the end of the positions produces nothing at all. The
    /// previous geometry then stands, which is the honest outcome — a mesh
    /// assembled from half a connectivity array is not a partial answer.
    #[test]
    fn out_of_range_indices_produce_nothing() {
        let mut inputs = quad();
        inputs.insert("indices", indices(&[0, 1, 9]));
        assert!(run(&request(inputs)).is_refusal());
    }

    #[test]
    fn an_unbound_required_input_produces_nothing() {
        let mut inputs = quad();
        inputs.remove("indices");
        assert!(run(&request(inputs)).is_refusal());

        // And the params are not consulted for any of this: a geometry filter
        // with nothing bound at all is inert rather than a panic.
        assert!(run(&request(HashMap::new())).is_refusal());
    }

    #[test]
    fn params_are_declared_for_every_input_the_run_reads() {
        for id in ["positions", "indices", "normals", "colour", "vertices"] {
            assert!(
                PARAMS.iter().any(|spec| spec.id == id),
                "the run reads \"{id}\" but no parameter declares it"
            );
        }
        // Guards against a required input losing its declaration, which would
        // make an unbound one reach `run` instead of being refused.
        let required: Vec<&str> = PARAMS
            .iter()
            .filter(|spec| spec.kind.is_required())
            .map(|spec| spec.id)
            .collect();
        assert_eq!(required, ["positions", "indices"]);
    }

    /// Nothing here reads a setting, so an empty map is complete. Worth
    /// asserting because `normalise` would otherwise quietly drop a parameter
    /// this filter had grown.
    #[test]
    fn the_kind_normalises_to_its_bindings_alone() {
        let mut registry = FilterRegistry::default();
        register(&mut registry);
        let kind = registry.get("geometry").expect("just registered");

        let mut given = ParamMap::default();
        given.insert("positions".into(), ParamValue::Data(7));
        let normalised = kind.normalise(&given);
        assert_eq!(normalised.get("positions"), Some(&ParamValue::Data(7)));
        assert_eq!(normalised.len(), 1, "{normalised:?}");
    }
}
