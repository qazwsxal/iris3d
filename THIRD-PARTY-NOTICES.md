# Third-party notices

iris3d includes and depends on work by other people. This file records what,
under which licence, and what each licence asks of us.

It exists because iris3d is going public under a split licence — free for
non-commercial use, paid for commercial use — and a licence of that shape can
only be offered over dependencies that permit it. Every entry below has been
checked against that constraint.

Generated against the state of the repository on 2026-08-14. Regenerate the
Rust half with:

```bash
cargo metadata --format-version 1
```

## Summary

Nothing in what iris3d ships is copyleft. The Rust dependency graph is 660
packages and every one of them is permissive — MIT, Apache-2.0, BSD, ISC, Zlib,
Unicode-3.0, BSL-1.0, CC0 or a dual licence offering one of those. The Python
runtime package depends on four things, all permissive.

Two obligations are real and ongoing:

1. **Keep the copyright notices.** MIT, BSD and Apache-2.0 all require the
   notice and the licence text to travel with any distribution, source or
   binary. That is the whole of what they ask.
2. **Do not let a copyleft dependency into the shipped set.** Two exist today and
   both are development-only. See [Not shipped](#not-shipped-development-only).

## Ported source

### Mol\*

`app/src/draw/cartoon.rs` follows Mol\*'s cartoon construction: the spline
interpolation and its tension handling from `mol-repr/.../curve-segment.ts`, and
the profile set, `aspectRatio` and `arrowFactor` controls and swapped nucleic
frame from `polymer-trace-mesh.ts`. The Rust is written fresh and the closed,
capped sweep is not Mol\*'s, but the design is theirs and the attribution is
owed either way.

- Project: Mol\* (molstar), <https://github.com/molstar/molstar>
- Licence: MIT
- Copyright (c) 2018-2024 Mol\* contributors

```
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

The cartoon construction itself — guide points from the trace atom, a direction
from the peptide plane, a spline through them — is Carson and Bugg's, published
in the *Journal of Molecular Graphics* in 1986. A published algorithm carries no
licence; the citation is here because it is the honest source, not because
anything requires it.

## Rust dependencies

Direct dependencies of `app`:

| Crate | Version | Licence |
|---|---|---|
| bevy | 0.19.0 | MIT OR Apache-2.0 |
| bevy_egui | 0.41.1 | MIT |
| clap | 4.5.60 | MIT OR Apache-2.0 |
| crossbeam-channel | 0.5.15 | MIT OR Apache-2.0 |
| futures | 0.3.32 | MIT OR Apache-2.0 |
| glob | 0.3.3 | MIT OR Apache-2.0 |
| log | 0.4.33 | MIT OR Apache-2.0 |
| prost | 0.14.3 | Apache-2.0 |
| tokio | 1.49.0 | MIT |
| tokio-stream | 0.1.18 | MIT |
| tonic | 0.14.5 | MIT |
| tonic-build | 0.14.5 | MIT |
| tonic-prost | 0.14.5 | MIT |
| tonic-prost-build | 0.14.5 | MIT |

The full graph, transitive dependencies included, is 660 packages under these
licences:

| Count | Licence |
|---|---|
| 330 | MIT OR Apache-2.0 |
| 123 | MIT |
| 47 | Apache-2.0 OR MIT |
| 24 | MIT/Apache-2.0 |
| 22 | Apache-2.0 |
| 22 | Unicode-3.0 |
| 16 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| 15 | Zlib OR Apache-2.0 OR MIT |
| 45 | the remainder: Zlib, BSD-2-Clause, BSD-3-Clause, ISC, BSL-1.0, CC0-1.0, MIT-0, Unlicense, 0BSD, and dual licences over them |

### Entries that need more than a notice

**`epaint_default_fonts` 0.35.0** — `(MIT OR Apache-2.0) AND OFL-1.1 AND
Ubuntu-font-1.0`. The crate's code is permissive; the `AND` covers the font
files it embeds, which arrive through `bevy_egui` and are in every build. Both
font licences permit embedding and redistribution, including in a commercial
product, and both require their own notice to travel along. They do **not**
propagate to iris3d's own code. Reserved font names must not be reused, which
matters only if a font is ever modified and reshipped.

**`self_cell` 1.2.2** — `Apache-2.0 OR GPL-2.0-only`. A choice, and iris3d takes
Apache-2.0. Recorded because the GPL option makes automated scanners flag it.

**`r-efi` 5.3.0** — `MIT OR Apache-2.0 OR LGPL-2.1-or-later`. As above; iris3d
takes MIT. UEFI bindings, pulled in by a target that is not built here.

**The 22 `icu_*`, `zerovec`, `yoke` and related crates** — `Unicode-3.0`. This
is the Unicode licence, which is permissive and requires the notice plus a
statement that the data is Unicode's. Reached through `idna` and the URL
handling below it.

## Python dependencies

The `iris3d` package installs four things:

| Package | Licence |
|---|---|
| grpcio | Apache-2.0 |
| numpy | BSD-3-Clause |
| protobuf | BSD-3-Clause |
| types-protobuf | Apache-2.0 |

All permissive. This is the set an end user actually receives.

### Not shipped: development only

`biotite` is a **development** dependency and is imported lazily, so installing
iris3d does not pull it in — see the module docstring in
`libraries/python/src/iris3d/molecules.py`, which records that this is
deliberate and names the reason:

> biotite is a *development* dependency, so it is imported lazily — installing
> iris3d does not drag in biotite (or its LGPL-licensed `biotraj` transitive
> dependency).

Two dev-only packages are not permissive:

- **`biotraj`** — LGPL. Reached through `biotite`. LGPL would oblige us to let a
  user relink it, which a commercial licence can accommodate but should not have
  to. It is not distributed.
- **`certifi`** — MPL-2.0. File-level copyleft, reached through `requests` under
  `pooch`. Not distributed.

Keeping both out of the shipped set is what the lazy import buys, and it is why
that import must stay lazy. If `biotite` ever becomes a runtime dependency, this
section has to be revisited before release, not after.

The remaining dev-only packages — `grpcio-tools`, `pooch`, `scikit-image`,
`trimesh`, `scipy`, `networkx`, `pillow`, `imageio`, `tifffile`, `msgpack`,
`requests`, `urllib3`, `idna`, `charset-normalizer`, `lazy-loader`,
`platformdirs`, `packaging`, `setuptools`, `typing-extensions` — are permissive
(Apache-2.0, BSD, MIT, HPND or PSF).

## Sample data

Nothing is vendored. `libraries/python/src/iris3d/scans.py` downloads on demand
and caches locally, so iris3d redistributes no dataset and the terms are the
source's own.

- **CTHead**, from the Stanford volume data archive,
  <https://graphics.stanford.edu/data/voldata/>. Fetched by
  `scans.cthead()`.

Structures fetched by `molecules.fetch()` come from the RCSB PDB, which places
its holdings in the public domain (CC0).

## Reference documents

`ref/mboit-bevy-reference.md` and `ref/moment-backend-brief.md` are iris3d's own
notes. The technique they describe — moment-based order-independent transparency
— is from published papers by Münstermann, Krumpen, Klein and Peters. Cited in
those documents; no licence attaches to an algorithm.
