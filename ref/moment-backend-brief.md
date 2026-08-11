# Brief: the moment-transparency backend

Scope: build `experimental`, a third rendering backend for iris3d implementing
moment-based order-independent transparency.

`ref/mboit-bevy-reference.md` is the governing design document. It records
derivations and traps already agreed in design discussion, and it is
**authoritative over anything in this brief**. Read it before writing code. This
brief exists to place that reference inside iris3d's own architecture, which the
reference says nothing about.

---

## 1. What a backend is here

A backend is a whole rendering pathway: one pipeline, together with the actor
kinds built for it. Backends are **mutually exclusive** and chosen once at
launch with `--backend`. They never mix. Two techniques that composite
differently cannot share a frame correctly, so choosing once removes the whole
class of interop questions rather than answering them one at a time.

Do not propose a shared actor-drawing layer to avoid duplication between
backends. That duplication is accepted deliberately: how a dataset is best
mapped onto GPU primitives depends on the pipeline, which is the whole reason
actors belong to a backend.

### The code you will be working beside

- **`app/src/draw/mod.rs`** — the shared layer. `Dirty` invalidation grading,
  `mark`/`clear_dirty`, `bound()` for resolving a binding to an array, the
  colour maps, and the `Invalidate` → `Draw` → `Place` system sets. It draws
  nothing itself, and every backend uses it unchanged.
- **`app/src/draw/default/`** — the reference pathway. One `Mesh3d` and one
  `Material` per actor on Bevy's standard pipeline. **Read this first.** It is
  the smallest complete example of the shape a backend takes: a plugin, a set of
  `register` functions, an `invalidate` system and a `draw_*` system per kind.
- **`app/src/draw/solari/`** — the raytracing pathway, if it has landed by the
  time you start. Useful as a second example, especially for how a backend adds
  its own components to the viewport's camera.
- **`app/src/scene/`** — objects, actors, bindings, the registry. **It knows
  nothing about rendering and must stay that way.** Do not add a `draw`
  dependency to it.

### Wiring your backend in

1. Add `Backend::Experimental` to the enum in `draw/mod.rs`. Give it a `name()`
   arm and a `requires()` arm (see §3).
2. Add one match arm in `DrawPlugin::build` that adds your plugin.
3. Your plugin calls `registry.served_by("experimental")` and registers its
   kinds, then adds its systems into the `Invalidate`, `Draw` and `Place` sets.

That is the entire integration surface. If you find yourself editing `scene/`,
`grpc/` or `ui/`, stop and reconsider — the seam was built so a new pathway
needs none of them.

---

## 2. Actor kinds, and what `shared` means

Each kind declares `shared: bool`. `true` promises that the id means the same
physical thing under every pathway — as a raytraced mesh and a rasterised mesh
are one mesh. `false` marks a kind only this pathway can offer, and the UI and
`ListActorKinds` flag it so a script knows it is not portable.

- `surface` and `volume` stay **`shared: true`**. They are the same physical
  things reached by different transport. `volume` should drop `steps`, which is
  a raymarching concept, and gain whatever the moment pathway needs. A shared id
  with different parameters is expected and correct — clients ask
  `ListActorKinds` and read the parameters rather than assuming.
- A kind you cannot draw is simply **not registered**. The refusal already names
  the running backend.
- Genuinely pathway-specific kinds get **`shared: false`**. See §6.

---

## 3. Gate it at launch

The pathway needs `Features::FLOAT32_BLENDABLE` to blend into 32-bit float
targets. It is not in the WebGPU baseline and not universal across backends.

`app/src/draw/probe.rs` reads adapter features before the app is built, because
`RenderDevice` does not exist until `RenderPlugin` has initialised. Add the
moment requirement to `Backend::requires`.

**Refuse and exit if the feature is missing.** Never fall back silently. A quiet
substitution shows a picture that is wrong and says nothing about it, which is
worse than not starting. `ref/mboit-bevy-reference.md` §9 lists the fallbacks
that would be acceptable if one is ever wanted — fewer moments in `Rgba16Float`,
or a signed thickness buffer with analytic compositing — but the default answer
is to fail loudly.

---

## 4. Build order — follow it, do not reorder

This is `ref/mboit-bevy-reference.md` §11, restated so this brief stands alone.
Validate each step before starting the next.

1. **Signed thickness.** `k = 0` only, one convex mesh, a single `Rg32Float`
   target. One draw with `cull_mode: None`, branching on
   `@builtin(front_facing)`; back faces add `+F_k(z)`, front faces subtract.
   Additive blending performs the pairing, so no fragment needs to know its
   partner, and non-convex and nested meshes need no special handling. Verify
   against an analytic sphere. **All the render-graph plumbing gets built and
   debugged in this step**, against arithmetic simple enough to check by hand.
2. **Analytic convex compositing.** `T = exp(-sigma * thickness)`. For a single
   convex volume this is the exactly correct image. Keep it as the reference to
   diff every later step against. **Do not skip this step.** Without a reference
   image you cannot tell moment error from sign error. It is the step that looks
   skippable and is not.
3. **Four power moments.** Same scene; the result must match step 2 closely. If
   it does not, the bug is in the warp or the bias, not in the moments. The
   biasing vector and epsilon come from the Münstermann 2018 supplementary, and
   the published epsilons assume low overdraw — raise them for high depth
   complexity. Start at 4, not 8.
4. **Non-convex and nested meshes.** Where moments earn their place.
5. **Depth warping** with a real per-frame depth bound. Use a polynomial warp so
   `F_k` stays closed form. A logarithmic warp breaks it for this formulation.
6. **Light-space moment map** and `T_light`, reusing the accumulation code. Keep
   opaque occluders in a separate conventional shadow map — an opaque Dirac has
   very large absorbance and wrecks the Hankel conditioning.
7. **Froxel prefix integral** `G` and in-scattering.
8. **Trigonometric moments**, if blob content comes into scope.

---

## 5. Bevy 0.19 specifics

The pass structure is opaque → moment → resolve → merge. In Bevy terms that is a
custom render phase plus `ViewNode`s inserted into the `Core3d` subgraph after
the main opaque pass.

**Copy the shape from Bevy 0.19's `custom_render_phase` example.** The render API
changed and is heavier than older tutorials suggest — expect
`ViewSortedRenderPhases<P>`, `DrawFunctions<P>`, `SpecializedMeshPipelines`,
`ViewKeyCache`, `DirtySpecializations`, `PendingCustomMeshQueues`,
`RenderVisibleEntities` and `phase.add_retained(..)`. Do not work from
remembered signatures; read the example and the local `bevy_render` source.

Non-negotiables, before you write any target descriptor:

- **fp32 moments, never fp16.** `F_k(z_out) - F_k(z_in)` is a small difference of
  two O(1) values with `k` up to 7. Thin shells cancel catastrophically in half
  precision. The 16-bit quantization tables in the MBOIT paper apply to
  Dirac-style surface fragments, not to this formulation.
- Additive blending: `BlendComponent { src_factor: One, dst_factor: One,
  operation: Add }`.
- Both transparent passes use `depth_write_enabled: false` and
  `CompareFunction::Less` against the opaque depth buffer.
- The resolve pass reads the moment textures with `textureLoad`, never filtered.
  Filtering across pixels is only correct in the light-space shadow map.
- WGSL has no complex type. Represent trigonometric moments as `vec2<f32>` pairs
  and write your own complex multiply.
- Check `Limits::max_color_attachments` and
  `max_color_attachment_bytes_per_sample`. The bytes-per-sample limit bites
  first.

---

## 6. Later kinds, once transparency works

Both `shared: false`, because in each case the technique *is* the depiction
rather than one route to a shared one:

- **`blobs`** — gaussian splats depositing into the moment buffer per
  `ref/mboit-bevy-reference.md` §3.2. Prefer the trigonometric basis here; the
  `exp(-k² σ² / 2)` factor damps high orders for wide blobs automatically.
- **Opacity-optimised lines**, in the LineVis manner. Its global solve over the
  whole line set decides what to reveal, which makes it a different picture of
  the data rather than the same picture drawn another way.

A plain `lines` kind — polylines with per-vertex colour — would be `shared:
true`, but it does not exist under any backend yet, so it is not yours unless
you want it.

---

## 7. How to verify

1. **Step 1**: read the target back and check thickness through an analytic
   sphere against `2*sqrt(r² - d²)`. Do not judge it by eye.
2. **Step 2** produces the reference image. Every later step diffs against it.
3. **Step 3** must match step 2 closely on the same scene. A visible band of
   broken pixels means Cholesky lost positive definiteness — raise epsilon.
4. `cargo test` must pass throughout. The `default` and `solari` backends are not
   yours to change; if a test in either starts failing, you have reached into
   shared code that should have stayed shared.
5. The app must still launch on `--backend default` with no behaviour change.

---

## 8. Performance, when you get there

From `ref/mboit-bevy-reference.md` §10, because the intuitive answer is wrong:
do **not** optimise for memory bandwidth first. Additive blending is
read-modify-write at the ROPs, and with tiled rasterization the tile footprint
stays in L2, so traffic scales with pixels rather than fragments. High depth
complexity is the cache-friendly direction.

Real costs in order: the second geometry pass, ROP blend throughput,
reconstruction ALU in the resolve pass, and only then memory bandwidth.
