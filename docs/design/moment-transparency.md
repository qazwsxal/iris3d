# Moment-based order-independent transparency

The `default` backend. **Source of truth for the code:**
[`src/draw/default/mod.rs`](../../src/draw/default/mod.rs). Background
reading: `ref/mboit-bevy-reference.md`.

## What it does

Opaque geometry goes through Bevy's ordinary passes and is lit normally.

Anything that transmits does **not** blend. A closed mesh is treated as the
boundary of a solid absorbing light at a uniform rate, and a sampled grid as a
medium along the ray. What you see through them is the interior — thick parts
read dark, thin parts clear — and nested or overlapping ones compose correctly
whatever order they are drawn in.

Absorbance accumulates up to the **opaque depth**, so a density map in front of a
ribbon dims it and one behind does not. That is what lets a structure be shown
inside the map it was built from, which is the thing this project is for.

## Why not ordinary alpha blending

Alpha blending is multiplicative, so it depends on the order fragments arrive in.
Sorting triangles fixes it only for meshes that do not intersect and are not
nested — which is exactly the case scientific data does not obey.

Absorbance is additive, and addition does not care about order:

```text
A(z) = -ln T(z) = sum of the absorbance of everything in front of z
```

So depth-dependent absorbance is accumulated with an additive blend, then
transmittance `T = exp(-A)` is reconstructed in a second pass. No sorting, no
per-pixel lists, and a fixed cost per pixel.

This is why the pathway requires `FLOAT32_BLENDABLE`. Additive blending into a
32-bit float target is what the whole method rests on, and it is not in the
WebGPU baseline.

## The signed-prefix trick

A fragment must contribute using **only its own depth** — that is what makes the
accumulation order-independent. A back face does not know which front face it
pairs with, and must not need to.

With uniform extinction `sigma` the absorbance has a density rather than a spike,
so every moment of it has an antiderivative:

```text
dA/dw   = sigma * span   inside the mesh, 0 outside
F_k(w)  = sigma * span * w^(k+1) / (k+1)
```

where `w` is depth warped into `[0, 1]` across the bound in
`prepare::MomentBounds`.

An interior interval contributes `F_k(w_out) - F_k(w_in)`. So front faces add
`-F_k(w)`, back faces add `+F_k(w)`, and **the additive blend performs the
pairing on its own**. Non-convex and nested meshes need no special handling,
which is the whole point.

One draw does both signs: `cull_mode: None` and a branch on
`@builtin(front_facing)`.

## What the moments buy

`k = 0` alone gives the total absorbance along the ray. That is already exact for
any arrangement of pure absorbers in front of opaque geometry — however tangled,
because the opaque depth clamp truncates each interval in the right place.

Four moments describe *where along the ray* the absorbance sits, so the resolve
can ask for the absorbance in front of any depth instead of only the total. That
is what transparent geometry at an intermediate depth needs, and what emission
and in-scattering will need.

## Build status

Steps 1 to 4 of `ref/mboit-bevy-reference.md` §11: signed thickness, an analytic
reference, four power moments, and nested meshes.

The render-world half was validated against the closed form for a sphere — a
sphere of known absorbance rendered into an orthographic camera of its own and
compared per pixel against `exp(-sigma * 2*sqrt(r^2 - d^2))`. That check is not
in the tree; recover it from history if a later step needs a reference image to
diff against.

**Not yet done:** a non-linear warp (step 5 — the warp here is linear, the
cheapest polynomial that keeps `F_k` closed form), light-space moments (step 6),
in-scattering (step 7), trigonometric moments (step 8). Nor per-view culling, or
batching through a real phase item.

Two notes in the reference do not survive contact with Bevy 0.19 and are
corrected in the code — see `draw::default::pass` for the depth handling and
`draw::default::pipeline` for the depth comparison.

## What it draws

Six kinds, split by whether they transmit.

| Pass | Kinds |
|---|---|
| Opaque, ordinary passes | `surface`, `points`, `ball-and-stick`, `glycan` |
| Transmitting, into the moment buffer | `medium`, `volume` |

The opaque kinds write depth and are the thing absorbance is measured *in front
of*.

`surface` and `medium` are the same triangles making different claims. A surface
is lit and opaque; a medium says those triangles *bound* something light is
absorbed inside. Two kinds rather than one with a mode, because they go through
different passes and their parameters are disjoint — `sigma` means nothing to a
lit surface and `double_sided` nothing to a medium. See `draw::default::medium`,
which also says why transparency is a kind here when every other tool makes it a
property.

An opaque kind carries **no** `MomentVolume`, which means `place_volumes` does
not match it — each has its own placement system instead. Anything opaque added
later needs the same pair.

---

# `medium` and `surface`: why two kinds

**Source of truth for the code:**
[`src/draw/default/medium.rs`](../../src/draw/default/medium.rs).

## Transparency is a kind here, and a property everywhere else

Worth knowing, because it looks like a mistake. ParaView, PyMOL and ChimeraX all
make transparency an *opacity setting* on the ordinary surface representation;
none of them has a separate kind for it.

This is not that. An opacity slider blends a surface with what is behind it; a
medium integrates absorbance along the path *through* a body. That is a different
physical claim, it needs a closed mesh, and it is the thing iris3d exists to
compose correctly against a volume. Blender agrees it is a different thing rather
than a slider, and models it as a volume absorption shader on an object's
interior — requiring the same closed manifold mesh.

**The rule is the pass.** `surface` goes through Bevy's ordinary opaque pass with
a `StandardMaterial`; `medium` goes through the moment and shell passes with a
`MomentVolume`. A mode inside a kind is right when the pass is the same — `solid`
against `film` is exactly that — and wrong when it is not. Their parameters are
disjoint too: `sigma` means nothing to a lit surface and `double_sided` nothing
to a medium.

Same geometry, two different claims about what it is, so two names. A client
binding triangles gets what it asked for rather than something else.

## The name

A **medium** in the physical sense: light passing through it is absorbed along
the way. The word `volume` is spent on the grid actor, so `medium` is what is
left and what the physics calls it.

Not `solid`, which is backwards twice over: in ChimeraX `solid` is the **opaque**
filled style — the opposite of this — and in ordinary English a solid sounds like
something you cannot see through. The word survives where it is right, as this
kind's `mode`, where `solid` means a body with thickness against `film`, a
surface without one.

## Two ways of drawing, over each other

An actor can produce both halves of what a piece of glass looks like:

- the **interior**, always — a body absorbing at `sigma` per unit of path length,
  which makes thick parts read dark and thin parts clear;
- the **boundary**, when `shell` is on — a thin dielectric skin adding a
  Fresnel-weighted specular reflection and absorbing nothing.

Two passes over the same mesh. The second is what stops the shape reading as
coloured fog.

There is no `double_sided`, unlike `surface`: it is a lighting choice, and both
faces are always drawn — they *have* to be, since the two of them are the
endpoints of the interval being integrated.

## Whether the mesh must be closed depends on `mode`

**In `solid` it must.** Every ray entering the interior has to leave it, or the
contributions do not cancel. The pathway cannot tell an open mesh from a closed
one: closedness is a property of the connectivity a client uploads, and checking
it would cost a pass over every edge on every rebuild to report a fact the client
already knows.

**In `film` it need not.** Each fragment is a spike at its own depth and needs no
partner, so an open shell, a lone triangle or a self-intersecting soup are all
valid. That is the mode for geometry you did not author — CAD tessellations
especially, which are routinely not closed. What it costs is thickness: every
crossing counts the same however deep the part is.

## What a medium does not read

The geometry's per-vertex **colours** mean nothing. Absorbance is a property of a
medium, so it is one value for the whole volume rather than something varying
across a surface the interior does not have. Colour arrives through the `tint`
parameter, read as a transmission.

Its **normals** are read, but only when a shell is on — the accumulation cares
where a boundary is, not which way it faces, so a volume with no skin never pays
for them.

Both are attributes of a mesh this kind shares rather than owns, so neither is
something it can decline to carry: the same geometry drawn as a lit `surface`
wants exactly the ones this pass ignores. What that costs is stride, and what it
buys is one upload instead of two. The accumulation pipeline pulls only the
position out of whatever layout it is given.
