# Moment-based order-independent transparency

The `default` backend. **Source of truth for the code:**
[`src/draw/default/mod.rs`](../../src/draw/default/mod.rs). The derivation is in
"The method in full" at the foot of this page.

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

Steps 1 to 4 of the build order below: signed thickness, an analytic
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

---

# The method in full

The governing derivation: where the closed forms come from, what the numbers have
to be, and what order to build the unbuilt parts in. The code above implements
steps 1 to 4 of the build order; the rest is written down so that picking it up
later does not mean rederiving it.

Primary source: Munstermann, Krumpen, Klein, Peters, *Moment-Based
Order-Independent Transparency*, I3D 2018 —
<https://momentsingraphics.de/I3D2018.html>. The supplementary has the biasing
vectors and quantization transforms, which are not reproduced here.

## Three content types

Every fragment must deposit its contribution using **only its own depth**. That
is the whole constraint; design any new content type to satisfy it.

**Surfaces** — thin, opaque-ish fragments. The absorbance measure is a Dirac
spike at the fragment depth:

```text
a = -log(1 - clamp(alpha, 0, ALPHA_MAX))
b_k += a * pow(z, k)          // power moments
b_k += a * cis(k * w * z)     // trigonometric moments
```

Clamp `alpha`; the logarithm diverges at 1.

**Gaussian blobs** — along the view ray the density is a 1D Gaussian and the
moment integral is closed form. Do not quadrature it. For ray `o + t*d` and
Gaussian `(m, Sigma)`:

```text
sigma_t2 = 1 / dot(d, Sigma_inv * d)
mu_t     = sigma_t2 * dot(d, Sigma_inv * (m - o))
```

`Sigma_inv` is already needed for the opacity evaluation — reuse it. Power
moments then follow a two-term recurrence, `mk = mu*m(k-1) + (k-1)*sigma2*m(k-2)`
from `m0 = 1`, `m1 = mu`. Trigonometric moments are one exponential,
`b_k = a * exp(i*k*w*mu - k*k*w*w*sigma2/2)`, whose damping factor suppresses
ringing for wide blobs automatically — prefer that basis here. Note that standard
3DGS discards depth extent after EWA projection; it has to be recovered with the
formulas above.

**Closed meshes with uniform interior absorbance** is the one this backend
implements, derived in "The signed-prefix trick" above.

## Depth warping

The warp maps view depth into a bounded moment domain, spending resolution where
fragments are. It **breaks the closed forms**: the integrals are over `z`, not
over `w(z)`.

- For Gaussians, linearise locally — `mu_w = w(mu)`, `sigma_w = |w'(mu)| * sigma`.
  Accuracy degrades for large blobs where the warp curves.
- For closed meshes, `F_k` must be the antiderivative of the *warped* power, so
  the warp has to be a polynomial for it to stay closed form. **Do not use a
  logarithmic warp here.** The code currently uses the cheapest polynomial — a
  linear one — which is step 5 of the build order below.

Set the bounds from a per-frame depth min/max. A global bound is acceptable at
first; refine it if contrast is poor.

## Reconstruction

For power moments:

1. Read the moment vector `b` and the total absorbance `b0`.
2. Normalise: `b_hat = b / b0`.
3. Bias: `b_biased = mix(b_hat, b_star, epsilon)`. **Without this the Hankel
   matrix loses positive definiteness and Cholesky fails**, showing as elongated
   bands of broken pixels.
4. Build the Hankel matrix, run Cholesky, solve for the polynomial coefficients.
5. Find the roots and evaluate the bound at the fragment depth.
6. Return `T = exp(-b0 * bound)`.

`b_star` and `epsilon` come from the MBOIT supplementary. The published `epsilon`
values assume low overdraw — raise them for high depth complexity. Unroll
everything in WGSL; the sizes are known at compile time.

The bias overestimates transmittance, so volumes read slightly too bright and
cores slightly too soft. Error peaks at *moderate* optical depth, not high.

## Volumetric shadows, and in-scattering

Neither is built. Both are the same problem as the view pass and reuse its code.

**Shadows.** Render the closed meshes from the light and accumulate
`+/-F_k(z_light)` with the identical sign rule; that gives a volumetric moment
shadow map to reconstruct `T_light` from. Keep **opaque** occluders in a separate
conventional shadow map and multiply the two transmittances — folding them into
the moment map destroys the Hankel conditioning, because an opaque Dirac has very
large absorbance. Moment shadow maps are prefilterable (blur, mip, MSAA them),
which is the main advantage over an alpha-blended deep shadow map.

**In-scattering.** The scattering integral does not decompose per fragment: a
back face knows `z_out` but not its matching `z_in`. Use the same prefix trick —
assume the medium fills all space, define

```text
G(z) = integral from z_near to z of  T_view(z') * T_light(z') * p(z') dz'
```

and an interior interval contributes `sigma_s * (G(z_out) - G(z_in))`, with the
signed accumulation restricting it to the true interior. `G` has no closed form,
so build it **per pixel, not per fragment**: march a froxel-style slice array
once per pixel, evaluate `T_view` from the view moments and `T_light` from the
shadow map at each slice, and bake the phase function there. Cost is
`O(pixels * slices) + O(fragments)` and does not scale with depth complexity,
which is what makes it affordable. Per-mesh scattering albedo is free —
`sigma_s` factors out linearly, so one prefix volume serves every mesh.

## Numerical requirements

**Use fp32 moments. Do not use fp16.** `F_k(z_out) - F_k(z_in)` is a small
difference of two O(1) values with `k` up to 7, and thin shells cancel
catastrophically in fp16. The 16-bit quantization tables in the MBOIT paper apply
to Dirac-style surface fragments, not to this formulation.

Keep the warped domain centred on zero, which reduces the dynamic range of the
high-order terms. Coloured extinction needs three independent moment sets —
budget for it before committing to a moment count.

## GPU specifics

`FLOAT32_BLENDABLE` is required and is not in the WebGPU baseline. This backend
refuses without it rather than degrading; the alternatives, if that is ever
revisited, are fewer moments in `Rgba16Float` at a precision cost, or a signed
thickness buffer with analytic compositing. **Do not silently produce wrong
images.**

- **Blend state:** `src_factor: One`, `dst_factor: One`, `operation: Add`.
- **Targets:** 8 power moments plus `b0` needs 2-3 `Rgba32Float` attachments.
  Check `max_color_attachments` and `max_color_attachment_bytes_per_sample` — the
  byte-per-sample limit bites first on mobile and some Vulkan drivers.
- **Depth:** both transparent passes use `depth_write_enabled: false` against the
  opaque depth buffer.
- **Front/back sign:** one draw, `cull_mode: None`, branch on
  `@builtin(front_facing)`.
- **No complex type in WGSL:** represent trigonometric moments as `vec2<f32>` and
  write your own complex multiply.
- **Bindings:** the resolve pass must not filter moments across pixels. Bind as
  `texture_2d<f32>` with `NonFiltering`, or `textureLoad` by integer coordinate.
  Filtering is only correct in the light-space shadow map.

Verify the Bevy render API against the crate version in use — the render graph,
`ViewNode` and view-uniform APIs change between minor releases. Two notes in the
original reference did not survive contact with Bevy 0.19; the corrections are in
`src/draw/default/pass.rs` (depth handling) and `src/draw/default/pipeline.rs`
(depth comparison).

## Performance model

The intuitive answer is wrong: **do not optimise for memory bandwidth first.**
Additive blending is read-modify-write at the ROPs, and with tiled rasterization
the tile footprint stays in L2, so traffic scales with pixels rather than
fragments. High depth complexity is the cache-friendly direction.

Real costs, in order:

1. **The second geometry pass** — vertex, primitive and tessellation throughput.
   This scales with scene complexity and no cache helps.
2. **ROP blend throughput** — wide fp32 blending has a fixed rate limit, and this
   does scale with fragment count.
3. **Reconstruction ALU** — Cholesky and root finding per fragment in the resolve.
4. Memory bandwidth, last.

The baseline to compare against is per-pixel linked lists (unbounded memory, can
fail to allocate) or depth peeling (one geometry pass per layer). A fixed bit
budget per pixel is what MBOIT buys against those.

## Build order

In sequence, validating each before continuing. **Steps 1 to 4 are done.**

1. **Signed thickness** — `k = 0` only, single convex mesh, `Rg32Float` target.
   Verify against an analytic sphere. Proves the sign rule and the pass structure.
2. **Analytic convex compositing** — `T = exp(-sigma * thickness)`, the correct
   answer for a single convex volume. Keep it as a reference image.
3. **4 power moments** — same scene, must match step 2 closely. If it does not,
   the bug is in the warp or the bias, not in the moments.
4. **Non-convex and nested meshes.** Now the moments earn their place.
5. **Depth warping** with a real per-frame depth bound.
6. **Light-space moment map** and `T_light`.
7. **Froxel prefix integral** `G` and in-scattering.
8. **Trigonometric moments**, if blob content comes into scope.

**Do not skip step 2.** Without a reference image you cannot tell moment error
from sign error.

How to check each: read the target back and compare thickness through an analytic
sphere against `2*sqrt(r^2 - d^2)` rather than judging by eye; diff every later
step against step 2's image; and treat a visible band of broken pixels as
Cholesky losing positive definiteness, which means raising epsilon. That
comparison is not in the tree — recover it from history if a later step needs it.

## Open questions

- Which warp keeps `F_k` closed form and still gives good near-field resolution?
  Try a low-order polynomial first.
- Do 4 power moments suffice for nested non-convex meshes, or are 6 needed?
  Measure against step 2's reference image.
- Not yet done beyond the build order: per-view culling, and batching through a
  real phase item.

## Further reading

- Peters, Klein. *Moment Shadow Mapping*. I3D 2015.
- Peters. *Non-Linearly Quantized Moment Shadow Maps*. HPG 2017.
- Kern et al. *A Comparison of Rendering Techniques for 3D Line Sets with
  Transparency*. TVCG 2020 —
  <https://www.willusher.io/publications/tvcg20_oit/>. Figure 1 documents
  MBOIT's failure modes honestly.
- NVIDIA `nvpro-samples/vk_order_independent_transparency` — seven OIT techniques
  toggled live on one scene; the best reference implementation to read.
- `chrismile/LineVis` — MBOIT alongside MLAB, MLAT, linked lists, depth peeling.
