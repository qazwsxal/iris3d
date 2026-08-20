# Why filters exist

**Source of truth for the code:** [`crates/iris3d-filter/src/lib.rs`](../../crates/iris3d-filter/src/lib.rs).

## The line

A **filter** reads arrays and parameters and writes arrays. An **actor** reads
arrays and draws them. Nothing straddles that line.

## The combinatorial argument

Without the split, every kind that *generates* geometry grows a mode for every
way of *displaying* it.

Concretely: a cartoon ribbon can be drawn as a lit surface or as an absorbing
medium. If one thing both builds the ribbon and draws it, it needs a `mode`
parameter that duplicates the entire difference between the `surface` and
`medium` actor kinds — inside a third place. Add a way of displaying, and every
generator grows another mode. Add a generator, and it has to implement every
mode.

That is `N * M`. Filters make it `N + M`: `N` ways to generate, `M` ways to
display, and any generated result can feed any display — or several at once. A
ribbon drawn as a surface *and* as a medium is built once and bound twice.

## Filters sit above the backends

A filter produces arrays, not GPU data, so it knows nothing about pipelines and
lives outside `draw` entirely. The same `contour` output feeds anything that can
draw triangles. This is why adding a rendering pathway does not touch a single
filter.

## How a filter takes part

A filter is an entity carrying the same components an actor does — a kind id, a
`ParamMap`, and `Bindings` derived from it — plus `Outputs`, mapping each
declared output to the handle it writes.

Those handles are allocated **when the filter is created** and are stable for its
life, so a client can bind an output before the first run has produced anything.
Each starts as an empty array in the `DataStore`.

## Chaining falls out of the asset system

A run rewrites its output asset **in place** rather than replacing the handle.
That raises `AssetEvent::Modified`, which is already what `draw::mark_dirty`
watches — so every actor bound to that handle re-dirties with no new code.

The same event marks a downstream *filter* stale. So filter A rewrites its
output, B is marked stale, B runs and rewrites its own, and the actor at the end
redraws. **Nothing walks a graph, and no filter knows its consumers.**

### The price: one frame per link

`AssetEvent::Modified` is not delivered until the frame after the write, so a
two-filter chain reaches the screen one frame later than a one-filter chain.

This is worth knowing and not worth removing. Removing it means walking the graph
in dependency order — which is exactly the coupling this shape exists to avoid —
to save a single frame on a chain that has already spent several frames running.

## Off the main thread

A run happens on `AsyncComputeTaskPool`, because the runs worth having are not
frame-sized: extracting a surface from a 256³ grid is not something to do between
two frames.

The cost is a copy. A task cannot borrow from the world, so it takes owned input
arrays — and a 256³ float grid is 64 MB. This is why `DataArray::clone` is
written to look like work rather than happening implicitly.

A run also carries the generation it started under. If the filter has gone stale
again since, the result describes inputs that no longer apply and is thrown away
rather than written — otherwise dragging a slider would leave whichever task
happened to finish last on screen.

## Outcomes report failure, not just absence

A run returns products **and** an optional problem, because "produced nothing" is
not one fact but several. A `cartoon` with no backbone, a `gather` handed indices
past the end of its values, and a `contour` whose level sits outside the field
are three different mistakes, and an empty output makes them look identical.

A problem does not mean nothing was produced, and products do not mean there was
no problem: a filter may emit what it can and still say that an input it wanted
was unusable. With arithmetic in the graph a length mismatch between two arrays
is the *routine* mistake, and the user needs to be told which two lengths rather
than left to guess why a wire went dead.

## Subsetting is filters, not a property of an actor

An actor is a dumb consumer of arrays: it reads what it is handed and puts it on
screen. Choosing *which* elements to hand it is a question about data, and
questions about data belong to filters.

A selection carried inline on an actor also could not be **wired**. It is not a
handle, so it appears nowhere in the graph, cannot be shared between two actors,
and cannot be computed from the data it selects over.

So subsetting is three filters, in `filter::index`:

- `gather` narrows per-element data — positions, elements, B-factors;
- `renumber` narrows connectivity;
- `reindex` re-densifies a hierarchy index whose numbering went sparse.

The selection itself comes from `subset`, over a mask built by `compare`, `logic`
and `match`. An actor sees only arrays that are already narrowed.

**The rule for connectivity** is VTK's extract-selection: an entry of a
connectivity array survives only when *every* element it names does. Keeping a
triangle with a dropped corner would mean inventing a position for it, and
clamping to a surviving neighbour draws a stretched sliver across the cut rather
than a clean boundary. `scene::subset::Remap` implements it.
