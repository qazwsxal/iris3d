# Contributing to iris3d

## Getting oriented

Read [README.md](README.md) first, then the header of
[`proto/iris3d/v1/scene.proto`](proto/iris3d/v1/scene.proto) — it explains the
data model, and the wire contract is the real interface.

The crate-level documentation in [`src/main.rs`](src/main.rs) maps the
Rust side. Generate and browse it:

```bash
cargo doc --no-deps --document-private-items --open
```

The two most likely first changes have walkthroughs:
[docs/adding-a-filter.md](docs/adding-a-filter.md) and
[docs/adding-an-actor-kind.md](docs/adding-an-actor-kind.md).

## Before opening a change

CI runs these; run them first.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
bash scripts/check-layering.sh
buf lint
```

The layering check is the one that needs explaining. The modules are ordered —
`data` and `model` below `scene`, `scene` below `filter`, `filter` below `draw`,
`draw` below `view` — and a module may only name the ones below it. Nothing in
the compiler enforces that, so the script does, by grepping `crate::` imports.
Comment lines are stripped before it looks, so a doc link pointing upward is
fine; only code makes a dependency.

If it fails, the fix is almost never to add an edge to its table. It is to move
the shared thing down to a layer both sides can see — that is how the periodic
table ended up in `data/` and `SceneError` in `model/`.

Documentation links are checked too, so a rename that breaks an intra-doc link
fails the build:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

Rendering is not covered by the tests. If you changed anything that reaches the
screen, look at it:

```bash
cargo run -- --screenshot out.png --screenshot-after 240
```

And run the Python demos against a started app, which exercise the gRPC surface,
the filters and the backend together:

```bash
cd libraries/python && uv run gallery_demo.py
```

## Comments

**Comments state what is true now.** History goes in commit messages, where it is
accurate and where git can find it.

This project has had to unpick a large amount of comment archaeology — prose
narrating what the code used to be, what was tried, and what was removed. It rots
silently, nothing checks it, and for a new reader it is worse than no comment
because it describes code that is not there.

So:

- **Do** explain *why* a non-obvious decision was made, in the present tense, as
  a constraint that still holds. "Two techniques that composite differently
  cannot share a frame correctly, so a backend is chosen once at launch" is
  worth its space.
- **Do not** write "this used to be X", "X has been removed", "this replaced Y",
  or a paragraph on an approach that was abandoned. If the old approach is a
  live risk of being reintroduced, state the constraint that rules it out, not
  the story.
- **Do not** leave a comment describing a field, module or count that a later
  change made wrong. Check the block above what you touched.
- Long design arguments belong in [docs/design/](docs/design/), where they can be
  maintained and linked, not spread across the definition sites of struct fields.

A module's `//!` header should be short enough that someone reads it. If it is
running past a screen, most of it belongs in `docs/design/`.

## Code layout

The modules are layered and work flows one way:

```
grpc  ──▶  scene  ──▶  filter  ──▶  draw  ──▶  screen
```

`scene` does not draw and does not know about gRPC. `filter` does not know about
a rendering pipeline. `draw` does not reach back up into the UI. If a change
needs a dependency that runs against that direction, it usually means something
is filed in the wrong module — raise it rather than adding the import.

## Commit messages

Since comments no longer carry history, commit messages have to. Explain what
changed and why it changed. If you removed an approach, say what it was and what
was wrong with it — that is exactly the text that should not go in a source file.
