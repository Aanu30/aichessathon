# Eagle Fang

A classical alpha-beta chess engine written for [AI Chessathon](https://aichessathon.com)
2026, sponsored by Optiver. One file, `agent.py`, exposing
`get_move(fen, time_left_ms) -> str`.

No neural network, no opening book, no third-party engine. The moves come from
search.

## The constraint that shapes everything

The match environment gives one core of a 2.60 GHz EPYC, 2 GB of RAM, no
network, and python-chess for board handling. Measured throughput is about
**25,000 nodes per second**. A C engine does millions.

Depth is therefore expensive, and anything that makes a single node count for
more is worth disproportionately more than it would be in a conventional
engine. Every choice below follows from that.

## Search

| Technique | Why it is here |
|---|---|
| Iterative deepening | Fills the transposition table and orders the next pass; guarantees a move when the clock runs out |
| Principal variation search | Null-window searches for every move after the first |
| Transposition table | Cuts re-search and supplies the best move for ordering |
| Aspiration windows | Narrow window around the previous score, widened on failure |
| Quiescence search | Captures only at the leaves, so evaluation is never measured mid-exchange |
| Null-move pruning | Guarded against zugzwang by requiring non-pawn material |
| Late-move reductions | Search moves late in the list shallower, re-search if they beat alpha |
| Futility and reverse futility | Skip quiet moves that cannot reach alpha at shallow depth |
| Late-move pruning | Stop examining a bad move list once deep into it |
| Check extensions | Never resolve a search while in check |
| Mate-distance pruning | Never look for a mate longer than one already found |

Move ordering, which is what makes alpha-beta pay: transposition-table move,
then captures by MVV-LVA with a static exchange evaluation to catch captures
that look good and lose material, then killer moves, counter-moves, and a
history heuristic.

## Evaluation

Tapered between midgame and endgame by the material remaining. Material and
piece-square scores are maintained **incrementally**, updated by the delta of
each move rather than rescanning the board, which would otherwise dominate the
cost of a node. Pawn structure is cached on the two pawn bitboards, which
change rarely.

Terms: material, piece-square tables, passed / doubled / isolated pawns,
bishop pair, rooks on open and semi-open files, king pawn shield and attacker
count, tempo.

## Time management

The budget comes from the clock actually handed to the agent, never from a
constant. The clock is checked inside the search every 1024 nodes, and
iterative deepening means aborting always leaves a completed move to return.

Measured worst case: 21% of the remaining clock consumed on a single move. It
still returns a legal move with 5 ms left.

## Results

Verified with the competition's own harness from
[`advitrocks9/aichessathon-starter`](https://github.com/advitrocks9/aichessathon-starter),
running Python 3.12.14 and python-chess 1.11.2 — the exact platform stack —
over the eight curated opening positions, each played once with each colour.

| Test | Result |
|---|---|
| `harness.arena` vs `baselines/greedy`, 16 games at 10s+0.1s | **+16 =0 -0**, every game by checkmate |
| `harness.arena` vs `baselines/minimax`, 16 games at 10s+0.1s | **+16 =0 -0**, every game by checkmate |
| `harness.package` | 11,893 byte zip, 44,052 unzipped, `agent.py` alone at the root, both smoke games pass |
| `ruff check` with the starter's rule set (E,F,I,N,UP,B,SIM,RUF) | clean |
| `mypy --strict` | clean |

Additionally, against local baselines: 40/40 against a random mover and 12/12
against a depth-3 alpha-beta with piece-square tables. Zero illegal moves,
zero crashes and zero flag falls across roughly 90 games. Worst case time
usage measured at 21% of the remaining clock on a single move.

Reaches depth 8-12 in five seconds depending on position, at roughly 25,000
nodes per second.

## Files

| File | |
|---|---|
| `agent.py` | The engine. This is the whole submission |
| `arena.py` | Local match harness with baseline opponents and a confidence interval. Not part of the submission |
| `RUNBOOK.md` | How to run it, what was fixed, and what to do next |

## Running it

```
pip3 install chess
python3 arena.py --games 12 --base 15 --inc 0.15 --opponent pst3
```

`mypy --strict` clean.
