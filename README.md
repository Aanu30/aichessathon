# Eagle Fang

A classical alpha-beta chess engine written for [AI Chessathon](https://aichessathon.com)
2026, sponsored by Optiver. One file, `agent.py`, exposing
`get_move(fen, time_left_ms) -> str`.

No neural network, no opening book, no third-party engine. The moves come from
search.

## Two engines, one file

The fast path is a **bitboard move generator and search jitted with numba**
(`fastcore.py`, `fastengine.py`), running at about **1.5 million nodes per
second**. The fallback is the original python-chess engine, at about 25,000.

The fast path is four to nine plies deeper for the same clock, so it plays.
The fallback exists because a numba compile failure at import would otherwise
lose every game, and a slow engine is infinitely better than no engine.

Selection happens once, at import, inside the 90 second init budget: the fast
engine is compiled, then verified against a known perft result. If either step
fails, `FAST_READY` stays False and the fallback plays. Every move the fast
path returns is also checked for legality against python-chess before it is
sent.

### Why the rewrite

python-chess caps the search at roughly 25,000 nodes per second, which is
depth 9 or 10 in a middlegame. Four evaluation improvements were built and
measured against that engine first; all four scored at or below 50% over 48
games each, and a control of the engine against a byte-identical copy of
itself scored 57%. Every effect was inside the noise. Depth was the only lever
large enough to see.

### Depth, measured

| Position | python-chess | numba | Gain |
|---|---|---|---|
| Opening | depth 10 in 2.9s | **depth 14 in 3.0s** | +4 plies |
| Middlegame | depth 8 in 2.4s | **depth 14 in 4.0s** | +6 plies |
| Rook endgame | depth 12 in 2.5s | **depth 21** | +9 plies |

### Design notes that mattered

**The whole search lives inside the jit boundary.** A Python-to-numba call
costs about 1.5 microseconds, worth roughly a hundred nodes here, so the
boundary is crossed once per depth iteration rather than once per node.

**Time is a node budget, not a clock check.** numba cannot read the clock
without an expensive object-mode escape, so Python measures the achieved node
rate, converts the remaining milliseconds into a node cap, and the search
aborts cleanly on reaching it.

**Compilation is warmed bottom-up.** Calling the search first makes numba
compile the whole call graph as one unit and LLVM's optimiser is superlinear
in function size: 56 seconds. Warming leaf functions first costs 26. IR-level
inlining is 10% faster at runtime but adds 25 seconds of compilation, so it is
deliberately not used. Init is about 25 seconds of the 90 allowed.

## Results

Verified with the competition's own harness from
[`advitrocks9/aichessathon-starter`](https://github.com/advitrocks9/aichessathon-starter),
under Python 3.12.14 with python-chess 1.11.2, numpy 2.5.2 and numba 0.67.0 --
the exact platform stack -- over the eight curated opening positions.

| Test | Result |
|---|---|
| **numba engine vs the python-chess engine** | **14-0, every game by checkmate**, both colours |
| Perft, six standard positions to depth 4-5 | all 26 correct, 8.7M nodes/sec |
| Legality over 394 random searched positions | zero illegal moves |
| `harness.package`, played out of the extracted zip | passes, init 23.4s |
| python-chess engine vs `baselines/greedy` | +16 =0 -0 |
| python-chess engine vs `baselines/minimax` | +16 =0 -0 |

On the six positions from the rounds 54 and 55 coaching report, the engine
that drew and lost those games played 0 of 6 correct moves and 5 of 6 named
blunders. This one plays 3 of 6 correct and 1 blunder -- with **no
passed-pawn term in its evaluation at all**. Depth fixed what evaluation
tuning could not.

## Files

| File | |
|---|---|
| `agent.py` | Entry point, the fast-path driver, and the python-chess fallback |
| `fastcore.py` | Bitboard move generation, jitted |
| `fastengine.py` | Search and evaluation, jitted |
| `regression.py` | Evaluation sanity checks plus six positions from real losses |
| `agent_v1_pythonchess_reference.py` | The previous submission, kept readable |
| `arena.py` | Local match harness with baseline opponents and a confidence interval. Not part of the submission |
| `RUNBOOK.md` | How to run it, what was fixed, and what to do next |

## Running it

```
pip3 install chess
python3 arena.py --games 12 --base 15 --inc 0.15 --opponent pst3
```

`mypy --strict` clean.
