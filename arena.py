"""Local arena: plays agent.py against baseline opponents under the real clock.

Not part of the submission. Run it with:

    python3 arena.py --games 20 --base 20 --inc 0.25

It enforces the same failure semantics the platform does: an illegal move, a
crash or a flag fall loses the game. It prints a score with a 95% interval so
you can tell a real improvement from noise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import random
import time
from typing import Callable, List, Optional, Tuple

import chess

MoveFn = Callable[[str, int], str]

# Eight roughly level opening positions, standing in for the curated set the
# platform uses. Tuning on these alone would overfit, so treat results as a
# sanity check rather than a rating.
OPENINGS: List[str] = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pp2pppp/2pp1n2/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/pp2pppp/8/2pp4/3P1B2/8/PPP1PPPP/RN1QKBNR w KQkq - 0 3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
]


def make_random(seed: int) -> MoveFn:
    rng = random.Random(seed)

    def move(fen: str, time_left_ms: int) -> str:
        board = chess.Board(fen)
        return rng.choice(list(board.legal_moves)).uci()

    return move


PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# A compact midgame piece-square table, mirrored for black. This baseline is
# roughly what a competent first attempt at the competition looks like: a
# fixed-depth alpha-beta over material plus position.
PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
PST_ROOK = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
PST_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
PST_KING = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 20, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
PSTS = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING,
}


def baseline_evaluate(board: chess.Board) -> int:
    score = 0
    for square, piece in board.piece_map().items():
        table = PSTS[piece.piece_type]
        value = PIECE_VALUE[piece.piece_type]
        if piece.color == chess.WHITE:
            score += value + table[square ^ 56]
        else:
            score -= value + table[square]
    return score if board.turn == chess.WHITE else -score


def make_pst_baseline(depth: int, seed: int) -> MoveFn:
    """Fixed-depth alpha-beta with MVV-LVA ordering. The realistic opponent."""
    rng = random.Random(seed)

    def negamax(board: chess.Board, d: int, alpha: int, beta: int) -> int:
        if board.is_checkmate():
            return -30000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if d == 0:
            return baseline_evaluate(board)
        moves = sorted(
            board.legal_moves,
            key=lambda m: (
                PIECE_VALUE.get(board.piece_type_at(m.to_square) or 0, 0) * 10
                - PIECE_VALUE.get(board.piece_type_at(m.from_square) or 0, 0)
            ),
            reverse=True,
        )
        best = -40000
        for move in moves:
            board.push(move)
            score = -negamax(board, d - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def move(fen: str, time_left_ms: int) -> str:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        rng.shuffle(moves)
        best_move = moves[0]
        best = -40000
        for candidate in moves:
            board.push(candidate)
            score = -negamax(board, depth - 1, -40000, -best)
            board.pop()
            if score > best:
                best = score
                best_move = candidate
        return best_move.uci()

    return move


def play_game(
    white: MoveFn,
    black: MoveFn,
    start_fen: str,
    base_seconds: float,
    increment: float,
    verbose: bool = False,
) -> Tuple[str, str, float, float]:
    """Return (result, reason, white_max_move_time, black_max_move_time)."""
    board = chess.Board(start_fen)
    clocks = {chess.WHITE: base_seconds, chess.BLACK: base_seconds}
    slowest = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    plies = 0

    while plies < 600:
        if board.is_game_over(claim_draw=True):
            break
        mover = white if board.turn == chess.WHITE else black
        side = board.turn
        start = time.monotonic()
        try:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                uci = mover(board.fen(), int(clocks[side] * 1000))
        except Exception as exc:
            return (
                "0-1" if side == chess.WHITE else "1-0",
                f"crash: {type(exc).__name__}: {exc}",
                slowest[chess.WHITE],
                slowest[chess.BLACK],
            )
        elapsed = time.monotonic() - start
        slowest[side] = max(slowest[side], elapsed)
        clocks[side] -= elapsed
        if clocks[side] <= 0:
            return (
                "0-1" if side == chess.WHITE else "1-0",
                "flag",
                slowest[chess.WHITE],
                slowest[chess.BLACK],
            )
        clocks[side] += increment
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return (
                "0-1" if side == chess.WHITE else "1-0",
                f"malformed: {uci!r}",
                slowest[chess.WHITE],
                slowest[chess.BLACK],
            )
        if move not in board.legal_moves:
            return (
                "0-1" if side == chess.WHITE else "1-0",
                f"illegal: {uci}",
                slowest[chess.WHITE],
                slowest[chess.BLACK],
            )
        board.push(move)
        plies += 1
        if verbose:
            print(f"  {plies:3} {uci} clock={clocks[side]:.1f}")

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "1/2-1/2", "ply cap", slowest[chess.WHITE], slowest[chess.BLACK]
    result = outcome.result()
    return result, outcome.termination.name, slowest[chess.WHITE], slowest[chess.BLACK]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--base", type=float, default=20.0, help="base seconds")
    parser.add_argument("--inc", type=float, default=0.25, help="increment seconds")
    parser.add_argument(
        "--opponent", choices=["random", "pst2", "pst3", "pst4"], default="pst3"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        import agent

    def me(fen: str, time_left_ms: int) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = agent.get_move(fen, time_left_ms)
        return result

    if args.opponent == "random":
        opponent = make_random(args.seed)
    else:
        opponent = make_pst_baseline(int(args.opponent[3:]), args.seed)

    wins = draws = losses = 0
    worst_ours = 0.0
    reasons: List[str] = []

    for game_index in range(args.games):
        # Reset per-game state the way a fresh process would.
        agent._SEARCHER = agent.Searcher()
        fen = OPENINGS[game_index // 2 % len(OPENINGS)]
        we_are_white = game_index % 2 == 0
        board_turn_white = chess.Board(fen).turn == chess.WHITE
        white, black = (me, opponent) if we_are_white else (opponent, me)
        result, reason, white_slow, black_slow = play_game(
            white, black, fen, args.base, args.inc
        )
        ours = white_slow if we_are_white else black_slow
        worst_ours = max(worst_ours, ours)
        if result == "1/2-1/2":
            draws += 1
            symbol = "="
        elif (result == "1-0") == we_are_white:
            wins += 1
            symbol = "+"
        else:
            losses += 1
            symbol = "-"
            reasons.append(reason)
        colour = "W" if we_are_white else "B"
        print(
            f"game {game_index + 1:3} as {colour} {symbol} {result:7} "
            f"{reason:22} slowest_move={ours:.2f}s "
            f"(start turn {'w' if board_turn_white else 'b'})",
            flush=True,
        )

    total = wins + draws + losses
    score = wins + 0.5 * draws
    rate = score / total if total else 0.0
    # Binomial standard error on the score rate, then a 95% interval.
    variance = (wins * (1 - rate) ** 2 + draws * (0.5 - rate) ** 2 + losses * rate**2)
    stderr = math.sqrt(variance / total) / math.sqrt(total) if total else 0.0
    low = max(0.0, rate - 1.96 * stderr)
    high = min(1.0, rate + 1.96 * stderr)
    print()
    print(f"+{wins} ={draws} -{losses}  score {score}/{total} = {rate:.3f}")
    print(f"95% interval [{low:.3f}, {high:.3f}]")
    if rate not in (0.0, 1.0) and 0 < rate < 1:
        elo = -400 * math.log10(1 / rate - 1)
        print(f"elo difference approx {elo:+.0f}")
    print(f"slowest move we played: {worst_ours:.2f}s")
    if reasons:
        print("loss reasons:", ", ".join(sorted(set(reasons))))


if __name__ == "__main__":
    main()
