# Sokoban solver — push-optimal A* per room

A solver for the make.fun Sokoban bounty (`sokoban-solver-01-run-20261121`):
one program answers a whole set of 5 levels in a single run.

## How it works

A generated level is a hallway with rooms off it. A box pushed out of a
room through its door can never be pushed back, so in an optimal answer
every box stays in its room — the level splits into independent rooms,
each solved push-optimally, and the answers are stitched together with
the player's walks between rooms.

Inside a room the search runs over **pushes**, not steps: a state is the
boxes' squares plus the player's reachable region (named by its lowest
square), and the cost is the number of pushes. A* with an admissible
estimate — the cheapest matching of boxes to distinct goals by
push-distance, ignoring the other boxes — finds a push-optimal answer.
Two prunings keep dead branches out: a box on a square from which no push
can ever reach a goal, and a 2 x 2 block of walls and boxes holding a box
off its goal (frozen). A breadth-first search over the same states, then
a greedy best-first search, back the A* up if it ever runs out of its
work budget.

Everything is budgeted by work counted, never by the clock, so two runs
on the same set write the same bytes.

## Results

On the kit's practice set the solver scores **1,000,000** (every instance
solved in exactly its push-optimal count P*), in about 0.2 s for the whole
set. It also scores 1,000,000 on 16 freshly drawn sets from the kit's
generator (random seeds), each in ~0.2 s — far inside the 60 s wall
budget for the set.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`). Node 22+
runs the TypeScript as it is; no dependencies, no network.

## License

MIT — see `LICENSE`.
