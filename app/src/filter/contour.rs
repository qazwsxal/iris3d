//! An isosurface through a scalar field, by Surface Nets.
//!
//! A field on a regular grid in, one mesh out. This is the filter that delivers
//! the `isosurface` representation: bind its geometry to [`surface`] for a lit
//! shape or to [`medium`] for one you can see the thickness of, and the
//! extraction happens once for both.
//!
//! [`surface`]: crate::draw::default::surface
//! [`medium`]: crate::draw::default::medium
//!
//! # Surface Nets, not marching cubes
//!
//! Both turn a sign change into triangles. They differ in where the vertices go:
//! marching cubes puts them *on the cell edges* and picks a triangulation from a
//! 256-entry table; Surface Nets puts **one vertex per cell**, at the average of
//! that cell's edge crossings, and joins the four cells around every crossed
//! edge into a quad. The connectivity is therefore uniform and needs no table.
//!
//! That suits what iris3d contours. Molecular surfaces and density blobs have no
//! sharp features, so Dual Contouring's QEF machinery — which exists to
//! reconstruct creases — buys nothing, and marching cubes' table plus the MC33
//! ambiguity fixes is complexity paid for a case that does not arise. Surface
//! Nets also gives better-shaped triangles and is **watertight by construction**,
//! which is exactly what `medium`'s signed prefix integral needs and what an
//! imported CAD tessellation does not give. See `iris3d-gpu-isosurface-options`.
//!
//! # This is the CPU reference
//!
//! Deliberately straightforward, and it runs on [`AsyncComputeTaskPool`] like
//! every other filter. A compute-shader version writing into the mesh's own
//! `MeshAllocator` slab is the intended end state — the reduction is enormous,
//! 16.7M cells in against ~10⁶ vertices out, which is where GPU extraction pays.
//! This exists so that version has something to be checked against.
//!
//! [`AsyncComputeTaskPool`]: bevy::tasks::AsyncComputeTaskPool
//!
//! # It colours its own output
//!
//! Unlike `cartoon`, which emits arrays for `colormap` to turn into colours
//! before `geometry` assembles them. A contour's values exist **only on the
//! surface it is building**: sampling `colour_field` where each vertex landed is
//! something only this run can do, so the ramp is read here, exactly as
//! [`volume`](crate::draw::default::volume) reads one per ray sample for the
//! same reason. Emitting a per-vertex scalar for `colormap` instead would mean
//! assembling the mesh a second time to attach the result.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::registry::{ParamKind, ParamSpec, float, text, uvec3, vec3, vector};

use super::colormap::{ColorMap, sample};
use super::{FilterKind, FilterRegistry, OutputKind, OutputSpec, Products, Request};

/// Ceiling on the samples one run will walk.
///
/// 256³ is 16.7 million and takes a noticeable fraction of a second on one
/// thread; 512³ is eight times that. Refusing beyond a bound is better than
/// spawning a task that never lands, because a filter that is still running is
/// indistinguishable from one that is merely slow.
const MAX_SAMPLES: usize = 300 * 300 * 300;

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "field",
        label: "field",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    // What to colour the surface by, sampled where each vertex lands. Unbound
    // means the surface takes the consumer's flat tint, which is right for a
    // contour of one quantity: every point on it has the same value by
    // construction, so colouring by the field itself would paint it one colour.
    ParamSpec {
        id: "colour_field",
        label: "colour by",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    // Absolute, in the field's own units, because that is what a threshold
    // means: "the 3-sigma contour" is a number, not a fraction of whatever
    // happens to be loaded. The range is wide enough not to clamp a real field,
    // which does make a slider over it useless — a normalised level beside this
    // one is the obvious thing to add when scrubbing matters.
    ParamSpec {
        id: "level",
        label: "level",
        kind: ParamKind::Float {
            default: 0.5,
            min: -1.0e6,
            max: 1.0e6,
            logarithmic: false,
        },
    },
    // The same three the `volume` actor declares, and meaning the same thing: a
    // 256³ grid states its geometry in nine numbers rather than 50 million
    // coordinates.
    ParamSpec {
        id: "dims",
        label: "samples",
        kind: ParamKind::Vector {
            components: 3,
            default: &[1.0, 1.0, 1.0],
            min: 1.0,
            max: 4096.0,
            integral: true,
        },
    },
    ParamSpec {
        id: "origin",
        label: "origin",
        kind: ParamKind::Vector {
            components: 3,
            default: &[0.0, 0.0, 0.0],
            min: -1.0e6,
            max: 1.0e6,
            integral: false,
        },
    },
    ParamSpec {
        id: "spacing",
        label: "spacing",
        kind: ParamKind::Vector {
            components: 3,
            default: &[1.0, 1.0, 1.0],
            min: 1.0e-6,
            max: 1.0e6,
            integral: false,
        },
    },
    ParamSpec {
        id: "map",
        label: "colour map",
        kind: ParamKind::Choice {
            options: super::colormap::MAPS,
            default: "viridis",
        },
    },
    // Equal ends autoscale over the bound field, spelled the same way the
    // `colormap` filter and the `volume` actor spell it so the three agree.
    ParamSpec {
        id: "range",
        label: "range (equal ends autoscale)",
        kind: ParamKind::Vector {
            components: 2,
            default: &[0.0, 0.0],
            min: -1.0e30,
            max: 1.0e30,
            integral: false,
        },
    },
];

const OUTPUTS: &[OutputSpec] = &[OutputSpec {
    id: "geometry",
    label: "surface",
    kind: OutputKind::Geometry,
}];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "contour",
        label: "isosurface",
        params: PARAMS,
        outputs: OUTPUTS,
        run,
    });
}

/// A field on a regular grid, indexed the way the wire delivers it.
struct Field {
    values: Vec<f32>,
    dims: UVec3,
}

impl Field {
    /// **z varies fastest**, which is what a numpy array of shape `(nx, ny, nz)`
    /// gives from a plain `.ravel()`. The same order `volume.rs` reads on the way
    /// to a 3D texture, and getting it wrong does not fail — it transposes the
    /// field, which looks like a plausible contour of the wrong thing.
    fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        let (ny, nz) = (self.dims.y as usize, self.dims.z as usize);
        self.values[(x * ny + y) * nz + z]
    }
}

fn run(request: &Request) -> Products {
    let mut products = Products::new();
    let Some(field) = request.input("field") else {
        return products;
    };
    let dims = uvec3(&request.params, "dims", UVec3::ONE);
    let expected = dims.x as usize * dims.y as usize * dims.z as usize;
    if expected == 0 || expected > MAX_SAMPLES {
        warn!("contour: {dims} samples is not a grid this will walk");
        return products;
    }
    let values = field.to_f32();
    if values.len() < expected {
        warn!(
            "contour: the field has {} values but {dims} needs {expected}",
            values.len()
        );
        return products;
    }
    // A contour needs a cell in every direction, so any axis of one sample has
    // no interior at all. Answering with nothing is right: a plane has no
    // isosurface, and inventing one from a single layer would be a guess.
    if dims.min_element() < 2 {
        debug!("contour: {dims} has no cells to walk");
        return products;
    }

    let field = Field {
        values: values[..expected].to_vec(),
        dims,
    };
    let level = float(&request.params, "level", 0.5);
    let surface = extract(&field, level, vec3(&request.params, "spacing", Vec3::ONE));
    if surface.positions.is_empty() {
        // The level is outside the field, or outside the part of it that has a
        // sign change. Producing nothing leaves whatever was there before, which
        // is the honest outcome for a slider dragged past the end of the data.
        debug!("contour: level {level} crosses nothing in this field");
        return products;
    }

    let origin = vec3(&request.params, "origin", Vec3::ZERO);
    let spacing = vec3(&request.params, "spacing", Vec3::ONE);
    let positions: Vec<[f32; 3]> = surface
        .positions
        .iter()
        .map(|p| (origin + *p * spacing).to_array())
        .collect();

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let vertices = positions.len();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        surface
            .normals
            .iter()
            .map(|n| n.to_array())
            .collect::<Vec<_>>(),
    );
    mesh.insert_indices(Indices::U32(surface.indices));

    if let Some(colours) = colours(request, &field, &surface.positions) {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    }

    debug!("contour: {vertices} vertices at level {level}");
    products.insert("geometry", mesh.into());
    products
}

/// Samples `colour_field` where each vertex landed and maps it through the ramp.
///
/// `None` when nothing is bound, when it is the wrong length for the grid, or
/// when the field is constant — a range of zero width has no reading, and
/// painting the whole surface with the bottom of the map would look like a
/// result rather than an absence of one.
fn colours(request: &Request, field: &Field, at: &[Vec3]) -> Option<Vec<[f32; 4]>> {
    let expected = field.values.len();
    let values = request.input("colour_field").map(|array| array.to_f32())?;
    if values.len() < expected {
        warn!("contour: the colour field is shorter than the grid");
        return None;
    }
    let source = Field {
        values: values[..expected].to_vec(),
        dims: field.dims,
    };

    let pinned = vector(&request.params, "range", 2);
    let sampled: Vec<f32> = at.iter().map(|p| trilinear(&source, *p)).collect();
    let (low, high) = match pinned[0] < pinned[1] {
        true => (pinned[0] as f32, pinned[1] as f32),
        false => span(&sampled)?,
    };

    let map = ColorMap::from_str(text(&request.params, "map", "viridis")).unwrap_or_default();
    let width = high - low;
    Some(
        sampled
            .iter()
            .map(|value| {
                let rgba = sample(map, (value - low) / width);
                [rgba[0], rgba[1], rgba[2], 1.0]
            })
            .collect(),
    )
}

/// The span of the values actually on the surface, or `None` if it has no width.
///
/// Over the sampled values rather than the whole field, which is the useful
/// autoscale: a potential that runs from -80 to +80 across a box may only reach
/// -5 to +5 on the surface being drawn, and stretching the ramp over the box
/// would leave the surface one flat colour.
fn span(values: &[f32]) -> Option<(f32, f32)> {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (low.is_finite() && high > low).then_some((low, high))
}

/// Reads a field at a fractional grid position, clamped to the grid.
fn trilinear(field: &Field, at: Vec3) -> f32 {
    let last = (field.dims - UVec3::ONE).as_vec3();
    let at = at.clamp(Vec3::ZERO, last);
    let base = at.floor();
    let frac = at - base;
    // The upper corner collapses onto the lower one at the far face, so the
    // weights there degenerate rather than reading off the end.
    let lo = base.as_uvec3();
    let hi = (lo + UVec3::ONE).min(field.dims - UVec3::ONE);
    let corner = |x: u32, y: u32, z: u32| field.at(x as usize, y as usize, z as usize);
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;

    let x0 = mix(
        mix(corner(lo.x, lo.y, lo.z), corner(lo.x, lo.y, hi.z), frac.z),
        mix(corner(lo.x, hi.y, lo.z), corner(lo.x, hi.y, hi.z), frac.z),
        frac.y,
    );
    let x1 = mix(
        mix(corner(hi.x, lo.y, lo.z), corner(hi.x, lo.y, hi.z), frac.z),
        mix(corner(hi.x, hi.y, lo.z), corner(hi.x, hi.y, hi.z), frac.z),
        frac.y,
    );
    mix(x0, x1, frac.x)
}

/// A surface in **grid** coordinates: positions are fractional sample indices,
/// so the caller applies the origin and spacing.
struct Surface {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<u32>,
}

/// The eight corners of a cell, as `(dx, dy, dz)` bit-encoded in that order:
/// corner `i` sits at `dx = i >> 2`, `dy = (i >> 1) & 1`, `dz = i & 1`.
const CORNERS: usize = 8;

/// The twelve edges of a cell, as pairs of [`CORNERS`] indices. Every pair
/// differs in exactly one bit, which is what makes it an edge.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along z
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along x
];

fn extract(field: &Field, level: f32, spacing: Vec3) -> Surface {
    let (nx, ny, nz) = (
        field.dims.x as usize,
        field.dims.y as usize,
        field.dims.z as usize,
    );
    // One cell per 2x2x2 block of samples, so one fewer along every axis.
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);

    let mut surface = Surface {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    };
    // Which vertex each cell produced, or `u32::MAX` for a cell the surface does
    // not pass through. A full grid rather than a map: the lookup happens four
    // times per crossed edge and is the inner loop of the second pass.
    let mut vertex = vec![u32::MAX; cx * cy * cz];
    let cell = |x: usize, y: usize, z: usize| (x * cy + y) * cz + z;

    for x in 0..cx {
        for y in 0..cy {
            for z in 0..cz {
                let mut values = [0.0f32; CORNERS];
                let mut inside = 0u8;
                for (index, value) in values.iter_mut().enumerate() {
                    let (dx, dy, dz) = (index >> 2, (index >> 1) & 1, index & 1);
                    *value = field.at(x + dx, y + dy, z + dz);
                    if *value >= level {
                        inside |= 1 << index;
                    }
                }
                // All in or all out: no crossing, so no vertex.
                if inside == 0 || inside == 0xFF {
                    continue;
                }

                // One vertex per cell, at the mean of that cell's crossings.
                // This is the whole of Surface Nets' placement rule, and it is
                // what gives it better-shaped triangles than marching cubes: the
                // vertex is free to sit anywhere in the cell rather than being
                // pinned to an edge.
                let mut sum = Vec3::ZERO;
                let mut crossings = 0.0f32;
                for (a, b) in EDGES {
                    let (va, vb) = (values[a], values[b]);
                    if (va >= level) == (vb >= level) {
                        continue;
                    }
                    let t = (level - va) / (vb - va);
                    sum += corner_offset(a).lerp(corner_offset(b), t);
                    crossings += 1.0;
                }
                if crossings == 0.0 {
                    continue;
                }

                vertex[cell(x, y, z)] = surface.positions.len() as u32;
                surface
                    .positions
                    .push(Vec3::new(x as f32, y as f32, z as f32) + sum / crossings);
                surface.normals.push(normal(&values, spacing));
            }
        }
    }

    // Every crossed edge between two samples joins the four cells around it into
    // a quad. That is the other half of Surface Nets, and it needs no table:
    // the connectivity is the same shape whatever the sign pattern was.
    for axis in 0..3usize {
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let here = [x, y, z];
                    let mut ahead = here;
                    ahead[axis] += 1;
                    if ahead[axis] >= [nx, ny, nz][axis] {
                        continue;
                    }
                    let low = field.at(here[0], here[1], here[2]);
                    let high = field.at(ahead[0], ahead[1], ahead[2]);
                    if (low >= level) == (high >= level) {
                        continue;
                    }

                    // The four cells sharing this edge, counter-clockwise seen
                    // from the +axis direction. `(b, c)` is the pair that makes
                    // `(axis, b, c)` right-handed, so the cycle below is the
                    // same three lines for all three axes.
                    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
                    if here[b] == 0 || here[c] == 0 {
                        continue;
                    }
                    let mut quad = [0u32; 4];
                    let steps = [(-1, -1), (0, -1), (0, 0), (-1, 0)];
                    let mut complete = true;
                    for (slot, (db, dc)) in steps.iter().enumerate() {
                        let mut coords = here;
                        coords[b] = (coords[b] as isize + db) as usize;
                        coords[c] = (coords[c] as isize + dc) as usize;
                        if coords[0] >= cx || coords[1] >= cy || coords[2] >= cz {
                            complete = false;
                            break;
                        }
                        let found = vertex[cell(coords[0], coords[1], coords[2])];
                        if found == u32::MAX {
                            // A crossed edge whose neighbouring cell produced no
                            // vertex cannot happen for a well-formed field, but
                            // a NaN makes every comparison false and gets here.
                            complete = false;
                            break;
                        }
                        quad[slot] = found;
                    }
                    if !complete {
                        continue;
                    }

                    // Normals point along **-gradient**, away from the high-value
                    // side, so a density blob gets outward-facing triangles. The
                    // winding has to agree: seen from +axis the cycle above is
                    // counter-clockwise, which is a front face exactly when the
                    // outward direction is +axis — that is, when the value falls
                    // as the edge is walked.
                    if low < high {
                        quad.reverse();
                    }
                    surface
                        .indices
                        .extend_from_slice(&[quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
                }
            }
        }
    }

    surface
}

/// Where a corner sits within its cell, in cell-local units.
fn corner_offset(index: usize) -> Vec3 {
    Vec3::new(
        (index >> 2) as f32,
        ((index >> 1) & 1) as f32,
        (index & 1) as f32,
    )
}

/// The outward normal of a cell, from the mean gradient across it.
///
/// Each component is the difference between the cell's two opposing faces,
/// averaged over their four corners — the gradient of the trilinear
/// reconstruction, averaged over the cell. Negated because the surface faces
/// *away* from the high-value side, and divided by the spacing because a
/// gradient in samples is not a gradient in world units: an anisotropic grid
/// tilts its normals, and a field sampled ten times more finely in z would
/// otherwise light as though it were squashed.
fn normal(values: &[f32; CORNERS], spacing: Vec3) -> Vec3 {
    let face = |mask: usize, shift: usize| {
        let mut low = 0.0;
        let mut high = 0.0;
        for (index, value) in values.iter().enumerate() {
            match (index & mask) >> shift {
                0 => low += value,
                _ => high += value,
            }
        }
        (high - low) / 4.0
    };
    let gradient = Vec3::new(face(0b100, 2), face(0b010, 1), face(0b001, 0)) / spacing;
    (-gradient).normalize_or(Vec3::Y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::registry::{ParamMap, ParamValue};
    use crate::scene::{DataArray, Dtype};
    use bevy::platform::collections::HashMap;

    /// A field sampled from `f` on an `n`-cubed grid centred on the origin, with
    /// unit spacing and the samples running 0..n.
    fn sampled(n: u32, f: impl Fn(Vec3) -> f32) -> DataArray {
        let mut values = Vec::with_capacity((n * n * n) as usize);
        // z fastest, as the wire delivers it.
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    values.push(f(Vec3::new(x as f32, y as f32, z as f32)));
                }
            }
        }
        DataArray::numeric(
            Dtype::Float32,
            vec![(n * n * n) as u64],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    fn centre(n: u32) -> Vec3 {
        Vec3::splat((n - 1) as f32 / 2.0)
    }

    /// A ball of radius `r`, as a field that is high inside and falls off
    /// outside — the shape of a density blob rather than a signed distance.
    fn ball(n: u32, r: f32) -> DataArray {
        let mid = centre(n);
        sampled(n, |p| r - p.distance(mid))
    }

    fn request(n: u32, field: DataArray, params: &[(&str, ParamValue)]) -> Request {
        let mut map = ParamMap::default();
        map.insert(
            "dims".into(),
            ParamValue::Vector(vec![n as f64, n as f64, n as f64]),
        );
        map.insert("level".into(), ParamValue::Float(0.0));
        for (id, value) in params {
            map.insert((*id).to_string(), value.clone());
        }
        let mut inputs = HashMap::new();
        inputs.insert("field", field);
        Request { params: map, inputs }
    }

    fn mesh(products: &Products) -> &Mesh {
        products["geometry"].geometry().expect("a mesh")
    }

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(values)) => {
                values.iter().map(|p| Vec3::from_array(*p)).collect()
            }
            _ => panic!("a mesh with no positions"),
        }
    }

    fn normals(mesh: &Mesh) -> Vec<Vec3> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(values)) => {
                values.iter().map(|n| Vec3::from_array(*n)).collect()
            }
            _ => panic!("a mesh with no normals"),
        }
    }

    /// The surface of a ball lands on the sphere it was asked for. This is the
    /// whole correctness claim: not "some triangles came out" but "they are
    /// where the level set is".
    #[test]
    fn a_ball_contours_to_its_own_radius() {
        let (n, r) = (24u32, 8.0);
        let products = run(&request(n, ball(n, r), &[]));
        let mesh = mesh(&products);
        let mid = centre(n);

        let radii: Vec<f32> = positions(mesh).iter().map(|p| p.distance(mid)).collect();
        assert!(!radii.is_empty(), "a ball should contour to something");
        let worst = radii
            .iter()
            .map(|found| (found - r).abs())
            .fold(0.0f32, f32::max);
        // Surface Nets averages a cell's crossings, so a vertex sits near rather
        // than exactly on the level set. Under a quarter of a sample is the
        // accuracy that buys the smoother triangles.
        assert!(worst < 0.25, "worst radius was off by {worst}");
    }

    /// Normals point out of a blob, not into it. Getting this backwards lights
    /// the surface from inside and shows as a shape that looks inside-out rather
    /// than as an error.
    #[test]
    fn normals_face_away_from_the_high_side() {
        let (n, r) = (24u32, 8.0);
        let products = run(&request(n, ball(n, r), &[]));
        let mesh = mesh(&products);
        let mid = centre(n);

        for (position, normal) in positions(mesh).iter().zip(normals(mesh)) {
            let outward = (*position - mid).normalize();
            assert!(
                normal.dot(outward) > 0.8,
                "normal {normal} faces the wrong way at {position}"
            );
        }
    }

    /// Watertight by construction, which is what `medium` needs: every edge of
    /// every triangle is shared by exactly one other triangle, walked in the
    /// opposite direction. A hole or a doubled face both show here.
    #[test]
    fn the_surface_is_closed_and_consistently_wound() {
        let (n, r) = (20u32, 6.0);
        let products = run(&request(n, ball(n, r), &[]));
        let mesh = mesh(&products);
        let Some(indices) = mesh.indices() else {
            panic!("a mesh with no triangles");
        };

        let corners: Vec<u32> = indices.iter().map(|i| i as u32).collect();
        let mut edges: std::collections::HashMap<(u32, u32), i32> = Default::default();
        for triangle in corners.chunks_exact(3) {
            for pair in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                // One counter per undirected edge, incremented one way and
                // decremented the other. A closed, consistently wound surface
                // leaves every one of them at zero.
                let key = (pair.0.min(pair.1), pair.0.max(pair.1));
                *edges.entry(key).or_default() += if pair.0 < pair.1 { 1 } else { -1 };
            }
        }
        let broken = edges.values().filter(|count| **count != 0).count();
        assert_eq!(broken, 0, "{broken} edges are not shared by two triangles");
    }

    /// Every edge used by exactly one triangle, with the vertex positions it
    /// runs between. An empty answer is a closed surface.
    fn open_edges(mesh: &Mesh) -> Vec<(Vec3, Vec3)> {
        let Some(indices) = mesh.indices() else {
            panic!("a mesh with no triangles");
        };
        let corners: Vec<u32> = indices.iter().map(|i| i as u32).collect();
        let mut uses: std::collections::HashMap<(u32, u32), usize> = Default::default();
        for triangle in corners.chunks_exact(3) {
            for pair in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                *uses
                    .entry((pair.0.min(pair.1), pair.0.max(pair.1)))
                    .or_default() += 1;
            }
        }
        let at = positions(mesh);
        uses.iter()
            .filter(|(_, count)| **count == 1)
            .map(|((a, b), _)| (at[*a as usize], at[*b as usize]))
            .collect()
    }

    /// A surface that leaves the grid is open **only where it leaves it**.
    ///
    /// This is the case that looks like a bug and is not: a lobe running out of
    /// the box is cut flat, and the cut reads as a hole. Worth pinning down,
    /// because a genuine hole in the middle of a surface looks exactly the same
    /// from a screenshot and means something entirely different — a medium drawn
    /// from an open mesh integrates the interval wrongly rather than merely
    /// looking odd.
    #[test]
    fn a_surface_is_open_only_where_it_leaves_the_grid() {
        // A ball too big for its box, so the level set is cut by all six faces.
        let (n, r) = (20u32, 14.0);
        let products = run(&request(n, ball(n, r), &[]));
        let open = open_edges(mesh(&products));
        assert!(!open.is_empty(), "this ball should be cut by the box");

        let last = (n - 1) as f32;
        for (a, b) in &open {
            // An open edge has to lie in one of the six boundary faces. Surface
            // Nets puts its vertices inside the cells, so the outermost one sits
            // half a cell in.
            let on_face = |p: Vec3| {
                p.min_element() <= 1.0 || (last - p.max_element()) <= 1.0
            };
            assert!(
                on_face(*a) && on_face(*b),
                "an edge from {a} to {b} is open away from the boundary"
            );
        }
    }

    /// The same field with room to close has no open edges at all — so the test
    /// above is measuring the boundary rather than tolerating a defect.
    #[test]
    fn a_surface_with_room_to_close_has_no_open_edges() {
        let (n, r) = (20u32, 6.0);
        let products = run(&request(n, ball(n, r), &[]));
        assert!(open_edges(mesh(&products)).is_empty());
    }

    /// A field with the shape iris3d actually contours — two lobes with a torus
    /// threaded through them — closes completely when it fits in its box.
    ///
    /// A ball is topologically the easy case. This is the one where a dual
    /// method that places a single vertex per cell would show tearing, at the
    /// waist where three sheets come close together.
    ///
    /// The level matters and is chosen, not guessed: at level 1 this same
    /// surface reaches the box and is cut by it, which produces 64 open edges
    /// that are not a defect. 25 closes it by about seven samples.
    #[test]
    fn a_lobed_field_closes() {
        let n = 40u32;
        let mid = centre(n);
        // The real 3d_z2 orbital: (3cos^2 - 1) * r^2 * exp(-r/3), squared.
        let field = sampled(n, |p| {
            let d = p - mid;
            let r = d.length().max(1.0e-6);
            let amplitude = (3.0 * (d.z / r).powi(2) - 1.0) * r * r * (-r / 3.0).exp();
            amplitude * amplitude
        });
        let products = run(&request(n, field, &[("level", ParamValue::Float(25.0))]));
        let mesh = mesh(&products);
        assert!(
            mesh.count_vertices() > 500,
            "the orbital should contour to a real surface, got {}",
            mesh.count_vertices()
        );

        let open = open_edges(mesh);
        assert!(
            open.is_empty(),
            "{} open edges, the first from {:?}",
            open.len(),
            open.first()
        );
    }

    /// The grid's spacing and origin place the surface in the world.
    #[test]
    fn the_grid_places_the_surface() {
        let (n, r) = (20u32, 6.0);
        let scaled = run(&request(
            n,
            ball(n, r),
            &[
                ("spacing", ParamValue::Vector(vec![2.0, 2.0, 2.0])),
                ("origin", ParamValue::Vector(vec![10.0, 0.0, 0.0])),
            ],
        ));
        let mid = centre(n) * 2.0 + Vec3::new(10.0, 0.0, 0.0);
        let radii: Vec<f32> = positions(mesh(&scaled))
            .iter()
            .map(|p| p.distance(mid))
            .collect();
        let worst = radii
            .iter()
            .map(|found| (found - r * 2.0).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.5, "worst radius was off by {worst}");
    }

    /// A level nothing reaches produces nothing rather than an empty mesh. The
    /// previous surface then stands, which is what a slider dragged past the end
    /// of the data should do.
    #[test]
    fn a_level_outside_the_field_produces_nothing() {
        let n = 16u32;
        let products = run(&request(
            n,
            ball(n, 5.0),
            &[("level", ParamValue::Float(1000.0))],
        ));
        assert!(products.is_empty());
    }

    /// A grid with no interior has no isosurface, and says so rather than
    /// indexing off the end of a single layer.
    #[test]
    fn a_grid_with_no_cells_produces_nothing() {
        let mut request = request(4, ball(4, 1.0), &[]);
        request
            .params
            .insert("dims".into(), ParamValue::Vector(vec![64.0, 1.0, 1.0]));
        assert!(run(&request).is_empty());
    }

    /// A field shorter than the grid it claims is refused rather than read past
    /// the end of.
    #[test]
    fn a_short_field_produces_nothing() {
        let mut request = request(8, ball(8, 3.0), &[]);
        request
            .params
            .insert("dims".into(), ParamValue::Vector(vec![16.0, 16.0, 16.0]));
        assert!(run(&request).is_empty());
    }

    /// A bound colour field is sampled where the vertices landed. A linear ramp
    /// in x should therefore come out varying across the surface rather than
    /// flat, and the extremes should reach the ends of the map.
    #[test]
    fn a_colour_field_is_sampled_on_the_surface() {
        let (n, r) = (20u32, 6.0);
        let mut request = request(n, ball(n, r), &[]);
        request.inputs.insert("colour_field", sampled(n, |p| p.x));

        let products = run(&request);
        let mesh = mesh(&products);
        let Some(bevy::mesh::VertexAttributeValues::Float32x4(colours)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("a bound colour field should have produced vertex colours");
        };
        let reds: Vec<f32> = colours.iter().map(|rgba| rgba[0]).collect();
        let low = reds.iter().copied().fold(f32::INFINITY, f32::min);
        let high = reds.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(high - low > 0.1, "the ramp did not vary: {low}..{high}");
    }

    /// Nothing bound to `colour_field` leaves the surface untinted, so the
    /// consumer's flat colour shows.
    #[test]
    fn an_unbound_colour_field_leaves_the_surface_plain() {
        let n = 20u32;
        let products = run(&request(n, ball(n, 6.0), &[]));
        assert!(mesh(&products).attribute(Mesh::ATTRIBUTE_COLOR).is_none());
    }

    /// The field is read with z varying fastest, as the wire delivers it.
    /// Reading it the other way does not fail — it transposes the surface, which
    /// looks like a plausible contour of the wrong thing.
    #[test]
    fn the_field_is_read_z_fastest() {
        // A step along x alone, so a transposed read puts the surface on the
        // wrong axis rather than merely moving it.
        let n = 8u32;
        let field = sampled(n, |p| 3.5 - p.x);
        let products = run(&request(n, field, &[]));
        let found = positions(mesh(&products));
        for position in &found {
            assert!(
                (position.x - 3.5).abs() < 0.6,
                "the surface should stand at x = 3.5, found {position}"
            );
        }
    }
}
