#!/usr/bin/env bash
# Fails if a module names one that sits above it.
#
# This replaces the compiler enforcement a crate-per-layer would give. The
# workspace split was tried and reverted: it cost a wider `pub` surface, broke
# every upward doc link, and bought about two seconds of incremental build time.
# This script buys the part that mattered — the layering — for twenty lines.
#
# Edges are declared bottom-up. A module may name anything listed for it and
# nothing else. Add an edge only when the dependency is one the layering says is
# allowed; if you find yourself wanting the reverse, that is the cycle this
# exists to catch.
set -uo pipefail

declare -A ALLOWED=(
  [bus]=""
  [counter]=""
  [redraw]=""
  [data]=""
  [model]="data"
  [scene]="bus counter redraw data model"
  [filter]="bus counter redraw data model scene"
  [draw]="bus counter redraw data model scene filter"
  [view]="bus counter redraw data model scene filter draw"
  [grpc]="bus counter redraw data model scene filter"
  # Wiring. `cli` parses argv and names nothing; `capture` writes screenshots and
  # needs only the redraw policy. `main` is exempt — adding the plugins in order
  # is exactly the job of naming every layer.
  [cli]=""
  [capture]="redraw"
)

ALL="bus counter redraw data model scene filter draw view grpc cli capture"
status=0

for module in "${!ALLOWED[@]}"; do
  target="src/$module"
  [ -d "$target" ] || target="src/$module.rs"
  [ -e "$target" ] || { echo "missing: $target"; status=1; continue; }

  # Comment lines are stripped: a doc link upward is allowed and is not a
  # dependency. Only code creates one.
  found=$(find "$target" -name '*.rs' -type f -exec cat {} + \
    | grep -vE '^\s*//' \
    | grep -oE "crate::($(echo "$ALL" | tr ' ' '|'))\b" \
    | sed 's/crate:://' | sort -u)

  for edge in $found; do
    [ "$edge" = "$module" ] && continue
    case " ${ALLOWED[$module]} " in
      *" $edge "*) ;;
      *) echo "layering: $module must not depend on $edge"; status=1 ;;
    esac
  done
done

if [ $status -eq 0 ]; then
  echo "layering: all module dependencies point downward"
fi
exit $status
