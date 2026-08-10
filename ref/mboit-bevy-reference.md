# Moment-Based OIT in Bevy / wgpu — Implementation Reference

Scope: order-independent transparency and volumetric shadows for closed meshes
with uniform interior absorbance. Includes notes for Gaussian-blob content.

Style note: this document is a working reference, not a tutorial. It records
derivations and traps agreed in design discussion. Verify all Bevy API names
against the version in `Cargo.toml` before you write code (see §9).

---

## 1. Core concept

Store the depth-dependent absorbance function `A(z)` per pixel as a small set of
moments. Moments accumulate additively. Additive accumulation is
order-independent. Reconstruct transmittance `T(z) = exp(-A(z))` from the
moments in a second pass.

Work in absorbance, not transmittance. Transmittance is multiplicative.
Absorbance is additive:

```
A(z) = -ln T(z) = sum over occluders in front of z of -ln(1 - alpha)
```

Two families of moments:

| Basis | Moment | Reconstruction | Use when |
|---|---|---|---|
| Power | `b_k = integral z^k dA` | Cholesky + root finding | Sharp depth features |
| Trigonometric | `b_k = integral exp(i k w z) dA` | Toeplitz / Levinson | Smooth media, blobs |

Power moments resolve steps better. Trigonometric moments are cheaper per
fragment and self-damp high frequencies. Prefer trigonometric moments for
volumetric and blob content. Prefer power moments for hard surfaces.

---

## 2. Pass structure

1. **Opaque pass.** Render opaque geometry. Keep the depth buffer.
2. **Moment pass.** Render all transparent geometry. Blend additively. Do not
   write depth. Test depth against the opaque buffer.
3. **Resolve / composite pass.** Render the same transparent geometry again.
   Reconstruct `T(z)` per fragment from the moment buffer. Blend
   `color * T(z) * alpha` additively.
4. **Merge.** Composite the accumulated colour over the opaque target.

The second geometry pass is the main cost. It doubles vertex and primitive work.
Profile geometry submission first if the renderer is slow. Do not assume the
moment buffer is the bottleneck (see §10).

---

## 3. Moment accumulation — three content types

Each fragment must deposit its contribution to `b_k` using only its own depth.
This is what makes the technique order-independent. Design every content type to
satisfy this rule.

### 3.1 Surfaces (thin, opaque-ish fragments)

The absorbance measure is a Dirac spike at the fragment depth.

```
a = -log(1 - clamp(alpha, 0, ALPHA_MAX))
b_k += a * pow(z, k)          // power moments
b_k += a * cis(k * w * z)     // trigonometric moments
```

Clamp `alpha`. The logarithm diverges at `alpha = 1`.

### 3.2 Gaussian blobs

Along the view ray the density is a 1D Gaussian. The moment integral is closed
form. Do not sample or quadrature it.

Get the ray-conditional mean and variance. For ray `o + t*d` and Gaussian
`(m, Sigma)`:

```
sigma_t2 = 1 / dot(d, Sigma_inv * d)
mu_t     = sigma_t2 * dot(d, Sigma_inv * (m - o))
```

`Sigma_inv` is already needed for the opacity evaluation. Reuse it.

Power moments of a normal distribution follow a two-term recurrence:

```
m0 = 1
m1 = mu
mk = mu * m(k-1) + (k-1) * sigma2 * m(k-2)
```

Trigonometric moments are one exponential:

```
b_k = a * exp(i*k*w*mu - k*k*w*w*sigma2 / 2)
```

The `exp(-k^2 sigma^2 / 2)` factor damps high orders for wide blobs
automatically. This is physically correct and suppresses ringing. Prefer the
trigonometric basis here.

Note: standard 3DGS discards depth extent after EWA projection. You must
recover it with the formulas above.

### 3.3 Closed meshes with uniform interior absorbance — PRIMARY TARGET

The measure has a density, not spikes. With uniform extinction `sigma`:

```
dA/dz = sigma  inside the mesh, 0 outside
```

Define an antiderivative and evaluate it at each fragment depth:

```
F_k(z) = sigma * pow(z, k+1) / (k+1)
```

Each interior interval contributes `F_k(z_out) - F_k(z_in)`.

**Sign rule.** Back-facing fragments add `+F_k(z)`. Front-facing fragments add
`-F_k(z)`. Verify the sign against your winding and depth convention on a single
cube before you trust it.

Implement with one draw. Set `cull_mode: None`. Read `@builtin(front_facing)` in
the fragment shader and negate.

No fragment needs to know its pairing partner. Additive blending performs the
pairing. Non-convex and nested closed meshes work without special handling.

`k = 0` reduces to `sigma * (z_out - z_in)`. That is a signed thickness pass. Use
it as your first milestone and validation target.

This content type is *easier* to reconstruct than surfaces. Surfaces make `A(z)`
a discontinuous staircase. Uniform interior absorbance makes it continuous and
piecewise linear. Try 4 power moments first. Do not start at 8.

**Required corrections:**

- **Near-plane clipping.** If the camera is inside the volume, front faces are
  missing. Add `-F_k(z_near)` once per pixel for that volume. Detect the case on
  the CPU or with a stencil count.
- **Opaque occluders.** Clamp the argument: use `F_k(min(z, z_opaque))`. Apply
  the same clamp to both endpoints. Intervals fully behind an occluder then
  collapse to zero. Additivity survives.

---

## 4. Depth warping

MBOIT maps view depth into a bounded moment domain. The warp spends resolution
where fragments are.

Warping breaks the closed forms in §3.2 and §3.3. The integrals are over `z`, not
over `w(z)`.

Two options:

- **Gaussians:** linearise locally. Use `mu_w = w(mu)` and
  `sigma_w = abs(w'(mu)) * sigma`. Accuracy degrades for large blobs where the
  warp curves. Accept the error or use a piecewise-linear warp.
- **Closed meshes:** `F_k` must be the antiderivative of the *warped* power.
  Choose a polynomial warp so the antiderivative stays closed form. Do not use
  a logarithmic warp here.

Set the warp bounds from a per-pixel or per-frame depth min/max. A global bound
is acceptable at first. Refine it later if contrast is poor.

---

## 5. Reconstruction

Steps for power moments:

1. Read the moment vector `b` and the total absorbance `b0`.
2. Normalise: `b_hat = b / b0`.
3. Bias: `b_biased = mix(b_hat, b_star, epsilon)`. `b_star` is the fixed biasing
   vector. Without this the Hankel matrix loses positive definiteness and
   Cholesky fails. Failure shows as elongated bands of broken pixels.
4. Build the Hankel matrix. Run Cholesky. Solve for the polynomial coefficients.
5. Find the polynomial roots. Evaluate the bound at the fragment depth.
6. Return `T = exp(-b0 * bound)`.

Get `b_star` and `epsilon` from the MBOIT supplementary document (Münstermann et
al. 2018). The published `epsilon` values assume low overdraw. Increase them for
high depth complexity.

Unroll everything in WGSL. Sizes are known at compile time.

The bias overestimates transmittance. Volumes read slightly too bright. Cores
read slightly too soft. Error peaks at moderate optical depth, not at high
optical depth.

---

## 6. Volumetric shadows

The light-space problem is the same problem. Reuse the code.

1. Render the closed meshes from the light. Accumulate `±F_k(z_light)` with the
   identical sign rule.
2. This gives a volumetric moment shadow map. Reconstruct `A_L(z)` and
   `T_light = exp(-A_L)` at any light-space depth.
3. Keep **opaque** occluders in a separate conventional shadow map. Multiply the
   two transmittances.

Do not fold opaque occluders into the moment map. An opaque Dirac has very large
absorbance. It destroys the conditioning of the Hankel matrix.

Moment shadow maps are prefilterable. Blur, mip, and MSAA them. That is the main
advantage over an alpha-blended deep shadow map.

If the light is inside a volume, apply the same implicit-boundary fix as §3.3.

---

## 7. In-scattering

The scattering integral does not decompose per fragment on its own. A back face
knows `z_out` but not its matching `z_in`.

Use the same prefix-integral trick. Assume the medium fills all space. Define:

```
G(z) = integral from z_near to z of  T_view(z') * T_light(z') * p(z') dz'
```

An interior interval then contributes `sigma_s * (G(z_out) - G(z_in))`. Signed
additive accumulation restricts it to the true interior.

`G` has no closed form. Build it per pixel, not per fragment:

- March a froxel-style slice array once per pixel.
- At each slice, evaluate `T_view` from the view moments and `T_light` from the
  shadow map.
- Bake the phase function `p(cos theta)` at each slice. You know the world
  position there. This handles point lights correctly.
- Slice in a warped depth so near-camera detail keeps resolution.

Cost is `O(pixels * slices) + O(fragments)`. It does not scale with depth
complexity. This is the reason the method is affordable.

Per-mesh scattering albedo is free. `sigma_s` factors out linearly. One prefix
volume serves every mesh.

---

## 8. Numerical requirements

**Use fp32 moments. Do not use fp16.**

`F_k(z_out) - F_k(z_in)` is a small difference of two O(1) values, with `k` up to
7. Thin shells cancel catastrophically in fp16. The 16-bit quantization tables in
the MBOIT paper apply to Dirac-style surface fragments, not to this formulation.

Keep the warped domain centred on zero. This reduces the dynamic range of the
high-order terms.

Colored extinction needs three independent moment sets. Budget for it before you
commit to a moment count.

---

## 9. Bevy / wgpu specifics

**Verify the Bevy render API against your actual crate version.** The render
graph, `ViewNode`, and view-uniform APIs change between minor releases. Do not
trust remembered signatures. Read the local docs or the `bevy_render` source.

**Blend state.** Additive accumulation:

```rust
BlendComponent {
    src_factor: BlendFactor::One,
    dst_factor: BlendFactor::One,
    operation: BlendOperation::Add,
}
```

**Critical feature flag.** Blending into 32-bit float render targets requires
`Features::FLOAT32_BLENDABLE`. It is not in the WebGPU baseline and it is not
universal across backends. Check for it at startup. Plan a fallback:

- Fall back to fewer moments in `Rgba16Float`, and accept the precision loss, or
- Fall back to a signed thickness buffer and analytic compositing (§11), or
- Fail loudly with a clear message. Do not silently produce wrong images.

**Targets.** 8 power moments plus `b0` needs 2 to 3 `Rgba32Float` attachments.
Check `Limits::max_color_attachments` and
`max_color_attachment_bytes_per_sample`. The byte-per-sample limit bites first on
mobile and on some Vulkan drivers.

**Depth.** Both transparent passes use `depth_write_enabled: false` and
`depth_compare: CompareFunction::Less` against the opaque depth buffer.

**Front/back sign.** One draw. `cull_mode: None`. Branch on
`@builtin(front_facing)`.

**No complex type in WGSL.** Represent trigonometric moments as `vec2<f32>`
pairs. Write your own complex multiply helper.

**Bindings.** The resolve pass samples the moment textures. Bind them as
`texture_2d<f32>` with `NonFiltering` sampling, or load by integer coordinate
with `textureLoad`. Do not filter moments across pixels in the view-space pass.
Filtering is only correct in the light-space shadow map.

---

## 10. Performance model

Do not optimise for memory bandwidth first. Additive blending is
read-modify-write at the ROPs. With tiled rasterization the tile footprint stays
in L2. Traffic scales with pixels, not with fragments. High depth complexity is
the cache-friendly direction.

Real costs, in order:

1. **Second geometry pass.** Vertex, primitive, and tessellation throughput. This
   is the cost that scales with scene complexity and no cache helps.
2. **ROP blend throughput.** Wide fp32 blending has a fixed rate limit. It does
   scale with fragment count.
3. **Reconstruction ALU.** Cholesky and root finding per fragment in the resolve
   pass.
4. Memory bandwidth. Usually last.

Compare against the right baseline. The alternatives are per-pixel linked lists
(unbounded memory, can fail to allocate) and depth peeling (one geometry pass per
layer). A fixed bit budget per pixel is the selling point of MBOIT against those.

---

## 11. Build order

Do these in sequence. Validate each before you continue.

1. **Signed thickness.** `k = 0` only, single convex mesh, `Rg32Float` target.
   Verify against an analytic sphere. This proves the sign rule and the pass
   structure.
2. **Analytic convex compositing.** `T = exp(-sigma * thickness)`. This is the
   correct final answer for a single convex volume. Keep it as a reference image.
3. **4 power moments.** Same scene. The result must match step 2 closely. If it
   does not, the bug is in the warp or the bias, not in the moments.
4. **Non-convex and nested meshes.** Now the moments earn their place.
5. **Depth warping** with a real per-frame depth bound.
6. **Light-space moment map** and `T_light`.
7. **Froxel prefix integral** `G` and in-scattering.
8. Trigonometric moments, if blob content is in scope.

Do not skip step 2. Without a reference image you cannot tell moment error from
sign error.

---

## 12. References

- Münstermann, Krumpen, Klein, Peters. *Moment-Based Order-Independent
  Transparency*. I3D 2018. Paper, supplementary, and video:
  <https://momentsingraphics.de/I3D2018.html>
  The supplementary has the biasing vectors and quantization transforms.
- Peters, Klein. *Moment Shadow Mapping*. I3D 2015.
- Peters. *Non-Linearly Quantized Moment Shadow Maps*. HPG 2017.
- Kern, Neuhauser, Maack, Han, Usher, Westermann. *A Comparison of Rendering
  Techniques for 3D Line Sets with Transparency*. TVCG 2020.
  <https://www.willusher.io/publications/tvcg20_oit/>
  Figure 1 documents MBOIT's failure modes honestly.
- NVIDIA `nvpro-samples/vk_order_independent_transparency` — seven OIT techniques
  toggled live on one scene. Best reference implementation to read.
- `chrismile/LineVis` — MBOIT alongside MLAB, MLAT, linked lists, depth peeling.

---

## 13. Open questions for implementation

- Which warp function keeps `F_k` closed form and still gives good near-field
  resolution? Try a low-order polynomial first.
- Does 4 power moments suffice for nested non-convex meshes, or is 6 needed?
  Measure against step 2's reference image.
- Is `FLOAT32_BLENDABLE` present on the target hardware? Decide the fallback
  before writing the resolve pass.
