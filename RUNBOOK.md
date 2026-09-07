# Eagle Fang — runbook

## 1. Upload now (2 minutes, no terminal)

`submission.zip` is built and smoke-tested. `agent.py` sits at the root of it,
which is the thing most people get wrong on their first upload.

1. Go to https://aichessathon.com/dashboard
2. Drag `submission.zip` onto the drop zone.
3. Wait for validation. It plays two smoke games and publishes a log.
4. Read that log. It prints your real init time and your time on every move —
   that is your only measurement of the match hardware, which is slower than
   your M4.

Rated rounds run hourly, 08:00–22:00. You get 10 uploads a day. Uploading a
working agent today costs nothing and starts collecting rating data.

## 2. Run it locally (terminal, copy-paste)

Open Terminal (Cmd+Space, type "Terminal", Enter). Paste one block at a time.

Install the one dependency:

```
pip3 install chess
```

Move into the folder with these files (drag the folder onto the Terminal
window after typing `cd ` and a space, then press Enter):

```
cd 
```

Play 12 games against a baseline opponent and print a score:

```
python3 arena.py --games 12 --base 15 --inc 0.15 --opponent pst3
```

Watch it think about one position:

```
python3 -c "import chess, agent; print(agent.get_move(chess.STARTING_FEN, 10000))"
```

Other opponents: `--opponent random`, `pst2`, `pst3`, `pst4`. Raise `--base` to
play at a slower time control. `arena.py` is a test harness only — it is not
part of the submission and must not go in the zip.

## 3. What the engine actually does

The whole design follows from one measured fact: python-chess gives about
**25,000 nodes per second** on this hardware. A C engine does millions. So
depth is expensive and everything that makes a single node count more is worth
disproportionately more than it would be in a normal engine.

**Search** — negamax with alpha-beta, wrapped in iterative deepening (search
depth 1, then 2, then 3…). Iterative deepening is not wasted work: each pass
fills the transposition table and gives the next pass good move ordering, and
it means there is always a finished move to hand back when the clock runs out.

**Move ordering** — the single biggest lever. Alpha-beta only prunes when good
moves come first. Order is: transposition-table move, then captures by
MVV-LVA (most valuable victim, least valuable attacker) with a static exchange
evaluation to catch captures that look good and lose material, then killer
moves, counter-moves, and a history table.

**Pruning** — null-move pruning, reverse futility, futility, late-move
reductions, late-move pruning. These buy depth that cannot be afforded
honestly. Each is guarded so it does not fire in check or in endings where
zugzwang makes the null-move assumption false.

**Quiescence search** — at the leaves, keep searching captures only. Without
it the evaluation gets measured halfway through an exchange and is simply
wrong.

**Evaluation** — tapered material and piece-square tables interpolated between
midgame and endgame by how much material is left, plus bishop pair, passed /
doubled / isolated pawns, rooks on open files, and a king safety term. The
material and piece-square part is **incremental**: it is updated by the delta
of each move instead of rescanning the board. Pawn structure is cached on the
two pawn bitboards, which change rarely.

**Clock** — budget is computed from the clock actually handed over, never from
a constant. The clock is checked inside the search every 1024 nodes. Worst
case measured: 21% of the remaining clock consumed on one move; it still
returns a legal move with 5ms left.

## 4. What was already fixed

The first version played illegal moves and lost 5 games out of 8. Cause: when
the time-out exception unwound out of the middle of the recursion, it left
moves pushed on the board, so the fallback picked a move that was legal in
some position deep inside the search tree. It now unwinds back to the real
position, and legality is checked against a fresh board built from the FEN.

Worth knowing because it is the failure mode the rules punish hardest — an
illegal move is an instant loss — and it only appears under time pressure,
which is exactly when local testing at generous time controls will not find it.

## 5. Verified against the official harness

The competition's own harness (`advitrocks9/aichessathon-starter`) was cloned
and run under Python 3.12.14 with python-chess 1.11.2 — the exact platform
stack, installed with uv — over the eight real curated opening positions.

| Test | Result |
|---|---|
| `make arena` vs `baselines/greedy` | +16 =0 -0, every game by checkmate |
| `make arena` vs `baselines/minimax` | +16 =0 -0, every game by checkmate |
| `make zip` (official packager and smoke) | passes; 11,893 bytes, 44,052 unzipped, `agent.py` alone at the root |
| `ruff check` with their rule set | clean |
| `mypy --strict` | clean |

Their ruff config is stricter than the default (E, F, I, N, UP, B, SIM, RUF at
line length 100). It flagged 70 issues, all style: pre-3.12 type annotation
syntax, and the nested clock check. Both fixed. Merging the clock check into
one condition costs nothing because `and` short-circuits, so `time.monotonic`
is still only called once every 1024 nodes.

The starter repo lives at `/tmp/starter` on this machine, with a working
Python 3.12 virtualenv at `/tmp/starter/.venv`. `/tmp` is cleared on reboot,
so fork the repo properly if you want to keep iterating with it.

## 5b. Honest assessment

Beating both shipped baselines 16-0 means the agent is a real entry and should
land comfortably inside the top 50. It does not mean it will win: the
baselines are meant to be beaten, and the field is other people's real
attempts, not `baselines/greedy`.

The ceiling is node throughput, and python-chess sets that ceiling at roughly
25,000 nodes per second. The teams that beat this will have written their own
bitboard move generator and jitted it with numba, which the platform
preinstalls for exactly this reason. That is worth roughly 50-100x the nodes,
which is about five extra plies.

That is the v2 decision, and it is a real one: a numba rewrite is a big job
with a lot of debugging, and a half-finished one that crashes scores zero
while this one scores points. Sequencing matters — ship this first.

Cheaper improvements, in rough order of value per hour:

1. Tune the search constants (null-move reduction, LMR amounts, futility
   margins) by running arena matches against the previous version. Keep every
   version as an opponent; "better than my last one" is the only comparison
   that counts.
2. Add 3- and 4-man syzygy endgame tablebases. They fit inside the 50 MB cap
   and end king-and-pawn races correctly, which a shallow search does not.
3. Retune the piece-square tables and evaluation weights.
4. The numba move generator.

Two games tell you nothing. A change worth 3% needs hundreds of games before
the interval shrinks past it. Note also that the eight openings are a sample,
not the published set, so anything tuned hard on them is tuned on eight
positions.

## 6. Separately: the Daily Five

From the rules page and the nav bar on your own dashboard. 6–10 September, one
attempt a day, five positions, 20 minutes, no engines. The **top 3 eligible UK
students each day win a seat at the London final** — a wildcard seat, not a
bracket place.

This is a completely separate route to London that depends on you playing
chess, not on your engine. It costs 20 minutes a day and you have not entered.
Do it.

## 7. Things you should be able to explain at the final

Finalists walk through how their agent was built. Answer these from memory,
not from the file:

1. Why does iterative deepening not waste the work of the shallower searches?
2. Why is move ordering worth more than a faster evaluation here?
3. What breaks if you delete the quiescence search?
4. When does null-move pruning give a wrong answer, and what guards against it?
5. Why is the evaluation updated incrementally instead of recomputed?
6. Why does the time budget come from `time_left_ms` rather than a constant?
