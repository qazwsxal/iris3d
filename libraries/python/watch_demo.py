"""Listening to what you do, and answering back.

The only call that streams both ways, and the only one where the *app* speaks
first. Everything else in the client asks a question and gets an answer; this
one waits.

Run it beside a loaded scene and click things::

    cargo run
    uv run python gallery_demo.py
    uv run python watch_demo.py

Ctrl-C to leave.

# Answering an event

The handler here does something rather than only printing: clicking an object
hides everything else, so the click has a visible consequence. That is the
pattern — **an event arrives on the stream, and the response goes out as an
ordinary call.** Nothing is pushed back up the stream, because everything worth
saying already has an RPC that validates it, and a second way to say it inside
the stream would be a second thing to keep correct.

A gRPC channel is thread-safe and multiplexes, so a call made from inside a
handler is another stream on the connection the events arrive on. Not a second
connection, and not queued behind the events.

# What this pattern is *not* for

High-rate events. A click is fine. A hover is not: out to Python, decide, call
back in, once per pointer move, is a round trip across a language boundary at
pointer rate. Feedback that is a pure function of what was picked belongs in the
graph instead — a `pick` node's output is an ordinary array, so a hover can
drive a subset with no client in the loop at all. That is the next piece of work
and this file is the argument for it.
"""

from __future__ import annotations

import sys

import iris3d


def main() -> int:
    with iris3d.Client(wait_timeout=iris3d.DEFAULT_CONNECT_TIMEOUT) as client:
        # Named up front, because an event carries handles and a handle is not
        # something anyone recognises.
        named = {obj.handle: obj.name for obj in client.list_objects()}
        print(f"watching {len(named)} objects - click one, Ctrl-C to stop")
        print("clicking tints what you hit; the one before it goes back")
        print()

        #: Linear RGB, as everything an actor takes is. See `draw::TINT`.
        HIGHLIGHT = (1.0, 0.45, 0.0)
        PLAIN = (0.8, 0.8, 0.85)

        lit: list[int] = []

        def respond(event: iris3d.PickEvent) -> None:
            """Tints the actor that was clicked, and un-tints the last one."""
            where = (
                " at ({:.1f}, {:.1f}, {:.1f})".format(*event.position)
                if event.position
                else ""
            )
            print(f"  [{event.object}] {named.get(event.object, '?')}{where}")

            # The response: ordinary calls, made from inside the handler. The
            # event named the *actor*, which is what has a tint — an object
            # drawn two ways is two actors and only one of them was clicked.
            #
            # `tint` is a parameter several kinds share and some do not have at
            # all; an unknown parameter is dropped rather than refused, so this
            # is safe to send at whatever was hit.
            for previous in lit:
                client.set_actor(previous, params={"tint": PLAIN})
            lit.clear()
            client.set_actor(event.actor, params={"tint": HIGHLIGHT})
            lit.append(event.actor)

        with client.watch_in_background(respond):
            # The main thread is free while the handler runs on its own. This
            # is the reason for the background form: a blocking `for event in
            # client.watch()` owns the thread it is on.
            try:
                while True:
                    input()
            except EOFError:
                pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
