"""Regression suite. Run this after every change to agent.py.

    python3 regression.py

Two kinds of check:

* Evaluation sanity. If the engine cannot say "I am a queen up" in numbers
  near 900, nothing downstream works, because the search has no gradient to
  climb between a won position and an equal one.
* Positions from real games the engine misplayed. Each names a move it must
  play and a move it must not. These came out of AI Chessathon rounds 54 and
  55, analysed at depth 20, where four won positions were drawn and a +2.8
  was converted into a loss.

Run at the match time control, not at infinite depth: what matters is what
the engine does with two seconds, not with two minutes.

Not part of the submission.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

import chess

# Importing the agent runs its warm-up search, which prints. Swallow that.
with contextlib.redirect_stdout(io.StringIO()):
    import agent

# (fen, must play, must not play, label)
POSITIONS: list[tuple[str, set[str], set[str], str]] = [
    (
        "4r3/5qk1/3p4/3P2p1/1p1P4/1P6/1P1Qp1PK/R7 b - - 9 51",
        {"Qh5+", "Kh6", "Kf6", "Kg6"},
        {"Qf4+"},
        "R54 m51 keep the queens on",
    ),
    (
        "4r3/4q3/3p2k1/3P2p1/1p1P4/1P4Q1/1P2p1PK/R7 b - - 1 47",
        {"e1=Q", "e1=R"},
        {"Qf6"},
        "R54 m47 promote the passer",
    ),
    (
        "8/8/3p4/3P1k2/1p1r1p2/1P6/1P2K1P1/4R3 b - - 0 57",
        {"Re4+"},
        {"Rxd5"},
        "R54 m57 activity over a pawn",
    ),
    (
        "8/8/8/8/1p2kp2/1P1pr3/1P3RP1/3K4 b - - 1 64",
        {"Ke5", "Rg3"},
        {"f3"},
        "R54 m64 do not liquidate",
    ),
    (
        "8/5kb1/3Pb3/6p1/4Pp2/2p5/p5PB/3R2K1 w - - 0 57",
        {"d7"},
        {"Re1"},
        "R55 m57 push the passer",
    ),
    (
        "5k2/3b2b1/2qP1p2/p3p1p1/1pp5/2P3B1/P1Q2PP1/3R2K1 w - - 0 42",
        {"f3"},
        {"Rd2"},
        "R55 m42 quiet consolidating move",
    ),
]

# (fen, minimum eval, maximum eval, label)
EVAL_CHECKS: list[tuple[str, int, int, str]] = [
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 800, 1100, "a queen up"),
    ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 420, 650, "a rook up"),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 60, 260, "a pawn up"),
    ("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1", 380, 1200, "passer on the 7th"),
    ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", 80, 320, "passer on the 4th"),
]


def check_evaluation() -> tuple[int, int]:
    print("Evaluation sanity")
    searcher = agent.Searcher()
    passed = 0
    for fen, low, high, label in EVAL_CHECKS:
        board = chess.Board(fen)
        searcher.set_position(board)
        score = searcher.evaluate(board)
        ok = low <= score <= high
        passed += ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label:22} {score:+6}  expected {low}..{high}")

    # The 7th-rank passer must be worth clearly more than the 4th-rank one.
    seventh = chess.Board(EVAL_CHECKS[3][0])
    fourth = chess.Board(EVAL_CHECKS[4][0])
    searcher.set_position(seventh)
    seventh_score = searcher.evaluate(seventh)
    searcher.set_position(fourth)
    fourth_score = searcher.evaluate(fourth)
    gap_ok = seventh_score - fourth_score >= 200
    print(
        f"  {'ok  ' if gap_ok else 'FAIL'} 7th-rank passer is worth "
        f"{seventh_score - fourth_score} more than a 4th-rank one (want >= 200)"
    )
    return passed + gap_ok, len(EVAL_CHECKS) + 1


def check_positions(budget_ms: int = 40_000) -> tuple[int, int, int]:
    print("\nPositions from rounds 54 and 55")
    good = 0
    forbidden = 0
    for fen, want, avoid, label in POSITIONS:
        board = chess.Board(fen)
        agent._SEARCHER = agent.Searcher()
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            uci = agent.get_move(fen, budget_ms)
        elapsed = time.perf_counter() - start
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  ILLEGAL {label}: {uci}")
            forbidden += 1
            continue
        san = board.san(move)
        if san in want:
            verdict = "PASS"
            good += 1
        elif san in avoid:
            verdict = "FAIL"
            forbidden += 1
        else:
            verdict = "----"
        print(
            f"  {verdict} {label:32} played {san:7} "
            f"want {'/'.join(sorted(want)):16} avoid {'/'.join(sorted(avoid)):6} "
            f"{elapsed:.1f}s"
        )
    return good, forbidden, len(POSITIONS)


def main() -> None:
    eval_passed, eval_total = check_evaluation()
    good, forbidden, total = check_positions()
    print(
        f"\nEvaluation {eval_passed}/{eval_total}. "
        f"Positions: {good}/{total} exact, {forbidden}/{total} played a forbidden move."
    )
    print("Baseline to beat, the build that played rounds 54 and 55: 0 exact, 5 forbidden.")
    if forbidden > 3:
        print("\nThis is worse than the build that drew and lost those games. Do not ship it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
