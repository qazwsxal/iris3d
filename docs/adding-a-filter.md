# Adding a filter

A **filter** reads arrays and parameters and writes arrays. It draws nothing and
knows nothing about a rendering pipeline. Narrowing a selection, extracting a
contour, building a ribbon and mapping values to colours are all filters.

[`crates/iris3d-filter/src/colormap.rs`](../crates/iris3d-filter/src/colormap.rs) is the smallest
complete example and the one to copy.

## The five pieces

### 1. Declare the parameters

A `&'static [ParamSpec]`. Each entry gives an id that goes over the wire, a label
the UI shows, and a `ParamKind` that decides both the control drawn for it and
its valid range.

```rust
const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "values",
        label: "values",
        kind: ParamKind::Array {
            dtypes: &[],       // empty accepts any numeric type
            shape: &[],        // empty accepts any shape
            required: true,
            structural: true,  // a change here rebuilds, rather than repaints
        },
    },
    ParamSpec {
        id: "map",
        label: "colour map",
        kind: ParamKind::Choice { options: FILTER_MAPS, default: "viridis" },
    },
];
```

Declare what you accept rather than inferring it from an array's name. A client
binds whatever it uploaded, under whatever name it chose, and the server tells it
what will fit.

`structural: true` means a change to this input invalidates generated geometry.
Leave it false for anything that only changes appearance — that is what keeps a
slider drag from rebuilding a mesh every frame.

### 2. Declare the outputs

A `&'static [OutputSpec]`. Each output gets a handle allocated **when the filter
is created**, so a client can bind it before the filter has ever run.

```rust
const OUTPUTS: &[OutputSpec] = &[OutputSpec {
    id: "colour",
    label: "colour",
    kind: OutputKind::Array { dtype: Some(Dtype::Float32), shape: &[0, 3] },
    provenance: Provenance::Identity("values"),
}];
```

`provenance` says how the output's elements relate to an input's — see
[`filter/provenance.rs`](../crates/iris3d-filter/src/provenance.rs). `Identity` means one
element out per element in, in order. Getting this right is what lets a pick in
the 3D view be traced back to the element it came from.

### 3. Write `run`

```rust
fn run(request: &Request) -> Outcome {
    let Some(values) = request.input("values") else {
        return Outcome::refused("has nothing bound to \"values\"");
    };
    let chosen = text(&request.params, "map", "viridis");
    // ...
    Outcome::from(products)
}
```

`run` executes on `AsyncComputeTaskPool`, off the main thread. It **cannot touch
the world** — its inputs are owned copies, which is a real cost for large arrays
and the reason `DataArray::clone` is written to look like work.

Read parameters through the accessors in
[`scene/registry.rs`](../crates/iris3d-scene/src/registry.rs) — `float`, `flag`, `text`,
`vector`, `vec3`, `uvec3` — rather than indexing the map. They fall back on a
default when a client sends nonsense, so a bad value gives a sensible result
instead of a panic.

Return `Outcome::refused(...)` rather than panicking or returning empty output.
A refusal is shown to the user and reported over the wire.

### 4. Register the kind

```rust
pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "colormap",
        label: "colour map",
        params: PARAMS,
        outputs: OUTPUTS,
        run: Some(run),
    });
}
```

`run: None` marks a **source** — a kind whose output changes when an event
happens rather than when its inputs do. [`filter/source.rs`](../crates/iris3d-filter/src/source.rs)
is the example; nothing schedules a source, something else writes its `Outputs`
directly.

### 5. Wire it in

Add the module to [`crates/iris3d-filter/src/lib.rs`](../crates/iris3d-filter/src/lib.rs) and call
its `register` beside the others:

```rust
pub(crate) mod yours;
// ...
yours::register(&mut registry);
```

## What you get for free

Chaining, staleness and redraw all fall out of the asset system. A run rewrites
its output asset in place rather than replacing the handle, which raises
`AssetEvent::Modified`, which marks every consumer stale — whether that consumer
is another filter or an actor. Nothing walks a graph and no filter knows who
reads it.

The price is **one frame per link**: a two-filter chain reaches the screen a
frame later than a one-filter chain.

## Testing

Filters are the easiest part of the codebase to test, because `run` is a plain
function from `&Request` to `Outcome` with no world involved. See the `mod tests`
blocks in [`filter/contour.rs`](../crates/iris3d-filter/src/contour.rs) — which counts
boundary edges to assert a surface is closed — and
[`filter/colormap.rs`](../crates/iris3d-filter/src/colormap.rs).

```bash
cargo test --workspace
```
