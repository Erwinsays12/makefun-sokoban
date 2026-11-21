// Sokoban solver: push-optimal A* search, one room at a time.
//
// A level is a hallway with rooms off it, each room holding its own boxes
// and goals, and a box never leaves its room in an optimal answer (a box
// pushed out through a door can never be pushed back). So the level splits
// into independent rooms, each solved push-optimally, and the answers are
// stitched together with the player's walks between rooms.
//
// Inside a room the search runs over pushes, not steps: a state is the
// boxes' squares plus the player's reachable region (named by its lowest
// square), and the cost is the number of pushes. A* with an admissible
// estimate — the cheapest way to match every box to a distinct goal by
// push-distance, ignoring the other boxes — finds a push-optimal answer.
// Two prunings keep dead branches out of the search: a box standing on a
// square from which no push can ever reach a goal, and a 2 x 2 block of
// walls and boxes holding a box off its goal (frozen). A breadth-first
// search over the same states, then a greedy best-first search, back the
// A* up if it ever runs out of its work budget.
//
// Everything is budgeted by work counted, never by the clock, so two runs
// on the same set write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four directions: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

// Work budgets (counts of expanded states), shared by the whole run so the
// total work stays bounded whatever the set holds.
const ASTAR_CAP = 400_000;
const BFS_CAP = 400_000;
const GREEDY_CAP = 400_000;
const TOTAL_CAP = 2_500_000;
let spent = 0;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: # wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The squares a box can stand on and still be pushed onto some goal, found
 * by pulling boxes back from the goals: a box at `p - d` can be pushed to
 * `p` when the square behind it, `p - 2d`, is open floor for the player.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const prev = p - d;
      const behind = prev - d;
      if (!wall[prev] && !wall[behind] && !live[prev]) {
        live[prev] = 1;
        queue.push(prev);
      }
    }
  }
  return live;
}

/**
 * For one goal, the fewest pushes that move a box from each square onto it,
 * ignoring the other boxes (a lower bound on the real cost).
 */
function pushDistances(level: Level, goal: number): Int32Array {
  const { w, wall } = level;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [goal];
  dist[goal] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const prev = p - d;
      const behind = prev - d;
      if (!wall[prev] && !wall[behind] && dist[prev] < 0) {
        dist[prev] = dist[p]! + 1;
        queue.push(prev);
      }
    }
  }
  return dist;
}

/**
 * The cheapest matching of boxes to goals by push-distance. With few boxes
 * every matching is tried; with many, each box takes its nearest goal (still
 * a lower bound, since one push moves one box).
 */
function estimate(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (const d of dists) if (d[b]! >= 0 && d[b]! < best) best = d[b]!;
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  // Branch and bound over the matchings, boxes in order.
  const used = new Array<boolean>(dists.length).fill(false);
  let best = Infinity;
  const walk = (i: number, cost: number): void => {
    if (cost >= best) return;
    if (i === n) {
      best = cost;
      return;
    }
    for (let j = 0; j < dists.length; j++) {
      if (used[j]) continue;
      const d = dists[j]![boxes[i]!]!;
      if (d < 0) continue;
      used[j] = true;
      walk(i + 1, cost + d);
      used[j] = false;
    }
  };
  walk(0, 0);
  return best;
}

/**
 * Flood the squares the player reaches from `from` around walls and the
 * boxes in `occupied`, stamping them in `seen`; returns the lowest square
 * reached, which names the region.
 */
function flood(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const q0 = p - w;
    const q1 = p + w;
    const q2 = p - 1;
    const q3 = p + 1;
    if (!wall[q0] && !occupied[q0] && seen[q0] !== stamp) {
      seen[q0] = stamp;
      stack[n++] = q0;
    }
    if (!wall[q1] && !occupied[q1] && seen[q1] !== stamp) {
      seen[q1] = stamp;
      stack[n++] = q1;
    }
    if (!wall[q2] && !occupied[q2] && seen[q2] !== stamp) {
      seen[q2] = stamp;
      stack[n++] = q2;
    }
    if (!wall[q3] && !occupied[q3] && seen[q3] !== stamp) {
      seen[q3] = stamp;
      stack[n++] = q3;
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of walls and boxes holding a box off its goal. */
function frozenBlock(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) {
      continue;
    }
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

interface SearchResult {
  pushes: number;
  moves: string;
}

/**
 * A push-optimal answer for `level`, or null when the work budget runs out.
 * `mode` picks the search order: 'astar' (pushes + estimate), 'bfs'
 * (pushes alone) or 'greedy' (estimate alone, not optimal, a last resort).
 */
function search(level: Level, cap: number, mode: 'astar' | 'bfs' | 'greedy'): SearchResult | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dists = goals.map((g) => pushDistances(level, g));
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The states, in the order found: their boxes, the player's square after
  // the push that made them, their parent and the push that made them.
  const nodeBoxes: number[][] = [level.boxes.slice()];
  const nodePlayer: number[] = [level.player];
  const nodeLow: number[] = [];
  const nodeG: number[] = [0];
  const nodeF: number[] = [0];
  const parent: number[] = [-1];
  const pushDir: number[] = [-1];
  const pushFrom: number[] = [-1];
  const seq: number[] = [0];

  const keyOf = (boxes: number[], low: number): string => String.fromCharCode(low, ...boxes);
  const best = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const low0 = flood(w, wall, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  nodeLow.push(low0);
  best.set(keyOf(level.boxes, low0), 0);
  nodeF[0] = mode === 'greedy' ? estimate(level.boxes, dists) : mode === 'astar' ? estimate(level.boxes, dists) : 0;

  // A binary heap of state indices on (f, -g, sequence): the smallest f
  // first, the deepest first on a tie, the earliest found first after that.
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean =>
    nodeF[a]! < nodeF[b]! ||
    (nodeF[a]! === nodeF[b]! && (nodeG[a]! > nodeG[b]! || (nodeG[a]! === nodeG[b]! && seq[a]! < seq[b]!)));
  const heapPush = (v: number): void => {
    let i = heap.length;
    heap.push(v);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const s = heapPop();
    if (best.get(keyOf(nodeBoxes[s]!, nodeLow[s]!)) !== s) continue;
    if (expanded >= cap || spent >= TOTAL_CAP) return null;
    expanded++;
    spent++;
    const boxes = nodeBoxes[s]!;
    const g = nodeG[s]!;
    for (const b of boxes) occupied[b] = 1;
    flood(w, wall, occupied, nodePlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let child: number[] | null = null;
        let key = '';
        let h = 0;
        if (!frozen) {
          child = boxes.slice();
          child[bi] = t;
          child.sort((x, y) => x - y);
          const low = flood(w, wall, occupied, b, mark, ++stamp, stack);
          key = keyOf(child, low);
          const prev = best.get(key);
          if (prev === undefined || nodeG[prev]! > g + 1) {
            h = mode === 'bfs' ? 0 : estimate(child, dists);
            if (h !== Infinity) {
              const idx = nodeBoxes.length;
              nodeBoxes.push(child);
              nodePlayer.push(b);
              nodeLow.push(low);
              nodeG.push(g + 1);
              nodeF.push(mode === 'greedy' ? h : g + 1 + h);
              parent.push(s);
              pushDir.push(k);
              pushFrom.push(b);
              seq.push(idx);
              best.set(key, idx);
              heapPush(idx);
              if (solved(child)) {
                found = idx;
                break;
              }
            }
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The search leaves `occupied` dirty on the push that found the answer
  // (its cleanup is skipped by the break), so clear it before rebuilding.
  occupied.fill(0);

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = nodeBoxes[s]!;
    const from = pushFrom[s]!;
    const k = pushDir[s]!;
    const d = DY[k]! * w + DX[k]!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH[k]!;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (up first on ties). */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds boxes is a room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DY[k]! * w + DX[k]!;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The level's moves: each room solved in turn, the other rooms' boxes standing as walls. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    // No rooms found: search the whole level as one.
    const whole = search(level, ASTAR_CAP, 'astar') ?? search(level, BFS_CAP, 'bfs') ?? search(level, GREEDY_CAP, 'greedy');
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  for (let r = 0; r < rooms.length; r++) {
    const room = rooms[r]!;
    const wall = level.wall.slice();
    for (let o = 0; o < rooms.length; o++) {
      if (o === r) continue;
      for (const b of done[o] ? rooms[o]!.goals : rooms[o]!.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved =
      search(sub, ASTAR_CAP, 'astar') ?? search(sub, BFS_CAP, 'bfs') ?? search(sub, GREEDY_CAP, 'greedy');
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done[r] = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
