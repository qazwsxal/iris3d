# Adding an actor kind

An **actor** is one way of drawing something. It binds the arrays it reads to the
inputs its kind declares, and it is drawn under the objects it is placed under.

Actor kinds belong to a **backend**, not to the shared layer. A backend is a
whole rendering pathway — one pipeline plus the kinds built for it — chosen once
at launch. How a dataset is best mapped onto GPU primitives depends on the
pipeline, which is why the kinds live inside
[`crates/iris3d-draw/src/default/`](../crates/iris3d-draw/src/default/) rather than above it.

[`draw/default/points.rs`](../crates/iris3d-draw/src/default/points.rs) is the smallest
complete example.

## The four pieces

### 1. A style component

The typed form of the kind's parameters, which the draw systems read instead of
searching a map.

```rust
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PointsStyle {
    pub size: f32,
    /// Linear RGB, used where nothing is bound to `colour`.
    pub tint: Vec3,
}
```

`PartialEq` matters: the component is rewritten from the parameter map on any
change, and Bevy's change detection is what decides whether the GPU data is
rebuilt.

### 2. Declare the parameters

As for a filter — a `&'static [ParamSpec]` naming inputs and settings. Two rules
worth stating:

**Take colours, not the numbers to make colours from.** An actor draws what it is
handed and chooses nothing. What ramp, over what range, is the `colormap`
filter's business, so colouring by anything a client can compute costs no change
in a kind.

**Take assembled geometry where you can.** `surface` and `medium` both bind a
mesh built by the `geometry` filter, so one set of vertex buffers is uploaded
once and referenced by both.

Shared specs live in [`draw/lib.rs`](../crates/iris3d-draw/src/lib.rs) — `iris3d_draw::TINT`
is one, so every kind spells its flat colour the same way.

### 3. Register the kind

`apply` writes the style component from a parameter map that is always complete
and in range, so it can read every declared parameter and expect it to be there.

```rust
pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "points",
        label: "points",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(PointsStyle {
                size: float(params, "size", 0.05),
                tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
            });
        },
    });
}
```

### 4. Wire it in

Add the module to [`draw/default/mod.rs`](../crates/iris3d-draw/src/default/mod.rs), call
its `register` beside the others, and add whatever systems build and draw its GPU
data.

```rust
mod yours;
// ...
yours::register(&mut registry);
```

## Rebuilds and repaints

[`draw::Dirty`](../crates/iris3d-draw/src/lib.rs) grades invalidation rather than treating
it as a single flag, because rebuilding geometry and changing a colour are not
the same cost. A kind's draw system reads the grade and does the least work that
covers it. This is what `structural: true` on a `ParamSpec` feeds.

## Shaders

Shaders are embedded in the binary, not loaded from an `assets/` directory — see
the comment on `SHADER` in
[`points.rs`](../crates/iris3d-draw/src/default/points.rs). Bevy resolves a filesystem
asset root that differs between `cargo run` and launching the executable
directly, and embedding removes the failure mode.

## Testing

Actor kinds are harder to test than filters, because drawing needs a world. Two
things are worth doing:

- Register the **real** `register` in the test app rather than a stand-in, so the
  declarations under test are the ones that ship. See the `mod tests` block in
  [`points.rs`](../crates/iris3d-draw/src/default/points.rs).
- Add the kind to [`draw/smoke.rs`](../crates/iris3d-draw/src/smoke.rs), which builds a
  world with every kind in it and checks nothing panics.

Then look at it, because a rendering change is only judgeable as an image:

```bash
cargo run -- --screenshot out.png --screenshot-after 240
```

Adding it to `libraries/python/gallery_demo.py` puts it on screen beside every
other kind, which is where a regression is most likely to be noticed.
