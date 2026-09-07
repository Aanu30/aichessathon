"""Eagle Fang - a classical alpha-beta chess engine for AI Chessathon.

Design notes
------------
The match environment gives one core of a 2.60 GHz EPYC and only python-chess
for board handling, so raw node throughput is low (tens of thousands of nodes
per second, not millions). Every design choice below follows from that:

* Evaluation is incremental. Material and piece-square scores are updated by
  the delta of each move rather than recomputed by scanning the board, which
  would otherwise dominate the cost of a node.
* Move ordering is the main lever. Alpha-beta only pays off when good moves
  come first, so we order by transposition-table move, then static exchange
  evaluation on captures, then killers, counter-moves and a history table.
* Forward pruning buys depth we cannot afford to search honestly: null-move
  pruning, reverse futility, futility, late-move reductions and late-move
  pruning.
* Quiescence search with delta pruning stops the evaluation being measured
  halfway through an exchange.
* The clock is checked inside the search, and iterative deepening means there
  is always a legal move to hand back when the budget runs out.

No third-party engine, network, or table of engine moves is used anywhere.
The piece-square tables are the well-known public PeSTO tuning constants.
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from typing import Callable, Dict, List, Optional, Tuple

import chess

# --------------------------------------------------------------------------
# Evaluation constants
# --------------------------------------------------------------------------

# Indexed by chess.PAWN..chess.KING (1..6); index 0 is unused padding.
MG_PIECE: Tuple[int, ...] = (0, 82, 337, 365, 477, 1025, 0)
EG_PIECE: Tuple[int, ...] = (0, 94, 281, 297, 512, 936, 0)

# Game-phase weight contributed by each piece type. Total for a full board
# is 24, which is what a tapered evaluation interpolates over.
PHASE_WEIGHT: Tuple[int, ...] = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24

# Piece values used by the static exchange evaluation. Flat, not tapered:
# SEE only needs a consistent ordering of the exchange.
SEE_VALUE: Tuple[int, ...] = (0, 100, 320, 330, 500, 900, 20000)

# Piece-square tables. Row 0 is rank 8 (a8..h8), so a white piece on square
# `sq` reads index `sq ^ 56` and a black piece reads index `sq`.
MG_PAWN = (
    0, 0, 0, 0, 0, 0, 0, 0,
    98, 134, 61, 95, 68, 126, 34, -11,
    -6, 7, 26, 31, 65, 56, 25, -20,
    -14, 13, 6, 21, 23, 12, 17, -23,
    -27, -2, -5, 12, 17, 6, 10, -25,
    -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22,
    0, 0, 0, 0, 0, 0, 0, 0,
)
EG_PAWN = (
    0, 0, 0, 0, 0, 0, 0, 0,
    178, 173, 158, 134, 147, 132, 165, 187,
    94, 100, 85, 67, 56, 53, 82, 84,
    32, 24, 13, 5, -2, 4, 17, 17,
    13, 9, -3, -7, -7, -8, 3, -1,
    4, 7, -6, 1, 0, -5, -1, -8,
    13, 8, 8, 10, 13, 0, 2, -7,
    0, 0, 0, 0, 0, 0, 0, 0,
)
MG_KNIGHT = (
    -167, -89, -34, -49, 61, -97, -15, -107,
    -73, -41, 72, 36, 23, 62, 7, -17,
    -47, 60, 37, 65, 84, 129, 73, 44,
    -9, 17, 19, 53, 37, 69, 18, 22,
    -13, 4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29, -53, -12, -3, -1, 18, -14, -19,
    -105, -21, -58, -33, -17, -28, -19, -23,
)
EG_KNIGHT = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25, -8, -25, -2, -9, -25, -24, -52,
    -24, -20, 10, 9, -1, -9, -19, -41,
    -17, 3, 22, 22, 22, 11, 8, -18,
    -18, -6, 16, 25, 16, 17, 4, -18,
    -23, -3, -1, 15, 10, -3, -20, -22,
    -42, -20, -10, -5, -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)
MG_BISHOP = (
    -29, 4, -82, -37, -25, -42, 7, -8,
    -26, 16, -18, -13, 30, 59, 18, -47,
    -16, 37, 43, 40, 35, 50, 37, -2,
    -4, 5, 19, 50, 37, 37, 7, -2,
    -6, 13, 13, 26, 34, 12, 10, 4,
    0, 15, 15, 15, 14, 27, 18, 10,
    4, 15, 16, 0, 7, 21, 33, 1,
    -33, -3, -14, -21, -13, -12, -39, -21,
)
EG_BISHOP = (
    -14, -21, -11, -8, -7, -9, -17, -24,
    -8, -4, 7, -12, -3, -13, -4, -14,
    2, -8, 0, -1, -2, 6, 0, 4,
    -3, 9, 12, 9, 14, 10, 3, 2,
    -6, 3, 13, 19, 7, 10, -3, -9,
    -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27,
    -23, -9, -23, -5, -9, -16, -5, -17,
)
MG_ROOK = (
    32, 42, 32, 51, 63, 9, 31, 43,
    27, 32, 58, 62, 80, 67, 26, 44,
    -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20,
    -36, -26, -12, -1, 9, -7, 6, -23,
    -45, -25, -16, -17, 3, 0, -5, -33,
    -44, -16, -20, -9, -1, 11, -6, -71,
    -19, -13, 1, 17, 16, 7, -37, -26,
)
EG_ROOK = (
    13, 10, 18, 15, 12, 12, 8, 5,
    11, 13, 13, 11, -3, 3, 8, 3,
    7, 7, 7, 5, 4, -3, -5, -3,
    4, 3, 13, 1, 2, 1, -1, 2,
    3, 5, 8, 4, -5, -6, -8, -11,
    -4, 0, -5, -1, -7, -12, -8, -16,
    -6, -6, 0, 2, -9, -9, -11, -3,
    -9, 2, 3, -1, -5, -13, 4, -20,
)
MG_QUEEN = (
    -28, 0, 29, 12, 59, 44, 43, 45,
    -24, -39, -5, 1, -16, 57, 28, 54,
    -13, -17, 7, 8, 29, 56, 47, 57,
    -27, -27, -16, -16, -1, 17, -2, 1,
    -9, -26, -9, -10, -2, -4, 3, -3,
    -14, 2, -11, -2, -5, 2, 14, 5,
    -35, -8, 11, 2, 8, 15, -3, 1,
    -1, -18, -9, 10, -15, -25, -31, -50,
)
EG_QUEEN = (
    -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30, 0,
    -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43, -5, -32, -20, -41,
)
MG_KING = (
    -65, 23, 16, -15, -56, -34, 2, 13,
    29, -1, -20, -7, -8, -4, -38, -29,
    -9, 24, 2, -16, -20, 6, 22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49, -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
    1, 7, -8, -64, -43, -16, 9, 8,
    -15, 36, 12, -54, 8, -28, 24, 14,
)
EG_KING = (
    -74, -35, -18, -18, -11, 15, 4, -17,
    -12, 17, 14, 17, 17, 38, 23, 11,
    10, 17, 23, 15, 20, 45, 44, 13,
    -8, 22, 24, 27, 26, 33, 26, 3,
    -18, -4, 21, 24, 27, 23, 9, -11,
    -19, -3, 11, 21, 23, 16, 7, -9,
    -27, -11, 4, 13, 14, 4, -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)

MG_TABLES: Tuple[Tuple[int, ...], ...] = (
    (), MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING,
)
EG_TABLES: Tuple[Tuple[int, ...], ...] = (
    (), EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING,
)


def _build_pst() -> Tuple[List[List[int]], List[List[int]]]:
    """Flatten the tables into [piece_type][square] lookups for each colour.

    Index as ``MG[colour][piece_type][square]``. Precomputing the mirror here
    keeps the hot path down to one list index instead of a conditional XOR.
    """
    mg: List[List[List[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
    eg: List[List[List[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
    for piece_type in range(1, 7):
        for square in range(64):
            mg[chess.WHITE][piece_type][square] = (
                MG_TABLES[piece_type][square ^ 56] + MG_PIECE[piece_type]
            )
            eg[chess.WHITE][piece_type][square] = (
                EG_TABLES[piece_type][square ^ 56] + EG_PIECE[piece_type]
            )
            mg[chess.BLACK][piece_type][square] = (
                MG_TABLES[piece_type][square] + MG_PIECE[piece_type]
            )
            eg[chess.BLACK][piece_type][square] = (
                EG_TABLES[piece_type][square] + EG_PIECE[piece_type]
            )
    # mypy-friendly flattening: indexed [colour][piece_type][square]
    return (
        [item for sub in mg for item in sub],
        [item for sub in eg for item in sub],
    )


_MG_FLAT, _EG_FLAT = _build_pst()
# _MG[colour * 7 + piece_type][square]
MG_PST: List[List[int]] = _MG_FLAT
EG_PST: List[List[int]] = _EG_FLAT

# Bitboard masks used by the pawn-structure and king-safety terms.
FILE_MASKS: Tuple[int, ...] = tuple(chess.BB_FILES)
ADJACENT_FILES: Tuple[int, ...] = tuple(
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
)


def _passed_masks(colour: chess.Color) -> Tuple[int, ...]:
    """Squares that must be free of enemy pawns for a pawn here to be passed."""
    masks: List[int] = []
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        span = 0
        ranks = range(rank_index + 1, 8) if colour == chess.WHITE else range(0, rank_index)
        for rank in ranks:
            span |= chess.BB_RANKS[rank]
        masks.append(span & (FILE_MASKS[file_index] | ADJACENT_FILES[file_index]))
    return tuple(masks)


PASSED_MASK: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
    _passed_masks(chess.BLACK),  # index 0 == chess.BLACK
    _passed_masks(chess.WHITE),
)

KING_ZONE: Tuple[int, ...] = tuple(
    chess.BB_KING_ATTACKS[square] | chess.BB_SQUARES[square] for square in range(64)
)

# Bonus for a passed pawn by the rank it has reached, from its own side's view.
PASSED_BONUS_MG: Tuple[int, ...] = (0, 5, 10, 20, 35, 60, 100, 0)
PASSED_BONUS_EG: Tuple[int, ...] = (0, 10, 20, 40, 70, 120, 180, 0)

BISHOP_PAIR_MG = 22
BISHOP_PAIR_EG = 45
DOUBLED_PAWN_MG = -8
DOUBLED_PAWN_EG = -22
ISOLATED_PAWN_MG = -12
ISOLATED_PAWN_EG = -14
ROOK_OPEN_FILE = 24
ROOK_SEMI_OPEN_FILE = 11
KING_SHIELD_BONUS = 12
KING_ATTACKER_PENALTY = -9
TEMPO_BONUS = 12

MATE_SCORE = 30000
MATE_THRESHOLD = 29000
INFINITY = 40000
MAX_PLY = 96

# Transposition-table entry flags.
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Search tuning constants.
NULL_MOVE_BASE_REDUCTION = 3
REVERSE_FUTILITY_MARGIN = 85
FUTILITY_MARGIN = 110
DELTA_MARGIN = 975
ASPIRATION_DELTA = 30
MAX_TT_ENTRIES = 900_000


def _make_position_key() -> Callable[[chess.Board], Hashable]:
    """Pick a way to identify a position for the transposition table.

    ``Board._transposition_key`` is private API. It exists in python-chess
    1.11.2, which is the version the platform pins, and it is by far the
    cheapest option. The Zobrist fallback exists so that a version change
    degrades performance instead of crashing on every node.
    """
    if hasattr(chess.Board, "_transposition_key"):
        return chess.Board._transposition_key

    # Bound under an alias: a plain ``import chess.polyglot`` here would make
    # `chess` a local name for this whole function and break the line above.
    import chess.polyglot as polyglot

    def zobrist(board: chess.Board) -> Hashable:
        return polyglot.zobrist_hash(board)

    return zobrist


POSITION_KEY = _make_position_key()


class TimeUp(Exception):
    """Raised inside the search when the move budget has been spent."""


class Searcher:
    """Holds everything that survives between moves in a single game."""

    def __init__(self) -> None:
        self.tt: Dict[Hashable, Tuple[int, int, int, Optional[chess.Move]]] = {}
        self.killers: List[List[Optional[chess.Move]]] = [
            [None, None] for _ in range(MAX_PLY)
        ]
        self.history: List[List[int]] = [[0] * 64 for _ in range(14)]
        self.counter_moves: Dict[Tuple[int, int], chess.Move] = {}
        # Pawn structure depends only on the two pawn bitboards, which change
        # rarely, so caching it turns the most expensive part of the leaf
        # evaluation into a dictionary lookup almost every time.
        self.pawn_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.game_keys: Dict[Hashable, int] = {}
        self.path_keys: List[Hashable] = []
        self.mg_stack: List[Tuple[int, int, int]] = []
        self.mg = 0
        self.eg = 0
        self.phase = 0
        self.nodes = 0
        self.deadline = 0.0
        self.stop = False
        self.root_best: Optional[chess.Move] = None
        self.seldepth = 0

    # -- incremental evaluation state --------------------------------------

    def set_position(self, board: chess.Board) -> None:
        """Recompute the incremental accumulators from scratch."""
        mg = 0
        eg = 0
        phase = 0
        for piece_type in range(1, 7):
            mg_w = MG_PST[chess.WHITE * 7 + piece_type]
            eg_w = EG_PST[chess.WHITE * 7 + piece_type]
            mg_b = MG_PST[chess.BLACK * 7 + piece_type]
            eg_b = EG_PST[chess.BLACK * 7 + piece_type]
            weight = PHASE_WEIGHT[piece_type]
            for square in chess.scan_forward(
                board.pieces_mask(piece_type, chess.WHITE)
            ):
                mg += mg_w[square]
                eg += eg_w[square]
                phase += weight
            for square in chess.scan_forward(
                board.pieces_mask(piece_type, chess.BLACK)
            ):
                mg -= mg_b[square]
                eg -= eg_b[square]
                phase += weight
        self.mg = mg
        self.eg = eg
        self.phase = phase
        self.mg_stack.clear()

    def push(self, board: chess.Board, move: chess.Move) -> None:
        """Play `move`, updating the incremental score by its delta."""
        self.mg_stack.append((self.mg, self.eg, self.phase))
        colour = board.turn
        sign = 1 if colour == chess.WHITE else -1
        from_square = move.from_square
        to_square = move.to_square
        piece_type = board.piece_type_at(from_square)
        if piece_type is None:  # pragma: no cover - defensive
            board.push(move)
            return
        base = colour * 7
        mg_own = MG_PST[base + piece_type]
        eg_own = EG_PST[base + piece_type]

        mg = self.mg
        eg = self.eg
        phase = self.phase

        # Remove the mover from its origin square.
        mg -= sign * mg_own[from_square]
        eg -= sign * eg_own[from_square]

        # Remove any captured piece.
        if board.is_en_passant(move):
            captured_square = to_square - 8 if colour == chess.WHITE else to_square + 8
            opp_base = (not colour) * 7
            mg += sign * MG_PST[opp_base + chess.PAWN][captured_square]
            eg += sign * EG_PST[opp_base + chess.PAWN][captured_square]
        else:
            captured = board.piece_type_at(to_square)
            if captured is not None:
                opp_base = (not colour) * 7
                mg += sign * MG_PST[opp_base + captured][to_square]
                eg += sign * EG_PST[opp_base + captured][to_square]
                phase -= PHASE_WEIGHT[captured]

        # Place the mover (or its promotion piece) on the destination square.
        if move.promotion:
            promoted = move.promotion
            mg += sign * MG_PST[base + promoted][to_square]
            eg += sign * EG_PST[base + promoted][to_square]
            phase += PHASE_WEIGHT[promoted]
        else:
            mg += sign * mg_own[to_square]
            eg += sign * eg_own[to_square]

        # Castling additionally shifts a rook.
        if piece_type == chess.KING and board.is_castling(move):
            if to_square > from_square:  # king side
                rook_from = chess.H1 if colour == chess.WHITE else chess.H8
                rook_to = chess.F1 if colour == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if colour == chess.WHITE else chess.A8
                rook_to = chess.D1 if colour == chess.WHITE else chess.D8
            mg_rook = MG_PST[base + chess.ROOK]
            eg_rook = EG_PST[base + chess.ROOK]
            mg += sign * (mg_rook[rook_to] - mg_rook[rook_from])
            eg += sign * (eg_rook[rook_to] - eg_rook[rook_from])

        self.mg = mg
        self.eg = eg
        self.phase = phase
        board.push(move)

    def pop(self, board: chess.Board) -> None:
        board.pop()
        self.mg, self.eg, self.phase = self.mg_stack.pop()

    # -- evaluation --------------------------------------------------------

    def pawn_structure(self, white_pawns: int, black_pawns: int) -> Tuple[int, int]:
        """Doubled, isolated and passed pawn terms, cached on the pawn skeleton."""
        cache_key = (white_pawns, black_pawns)
        cached = self.pawn_cache.get(cache_key)
        if cached is not None:
            return cached

        mg = 0
        eg = 0
        for file_index in range(8):
            file_mask = FILE_MASKS[file_index]
            white_on_file = chess.popcount(white_pawns & file_mask)
            black_on_file = chess.popcount(black_pawns & file_mask)
            if white_on_file > 1:
                mg += DOUBLED_PAWN_MG * (white_on_file - 1)
                eg += DOUBLED_PAWN_EG * (white_on_file - 1)
            if black_on_file > 1:
                mg -= DOUBLED_PAWN_MG * (black_on_file - 1)
                eg -= DOUBLED_PAWN_EG * (black_on_file - 1)
            neighbours = ADJACENT_FILES[file_index]
            if white_on_file and not (white_pawns & neighbours):
                mg += ISOLATED_PAWN_MG * white_on_file
                eg += ISOLATED_PAWN_EG * white_on_file
            if black_on_file and not (black_pawns & neighbours):
                mg -= ISOLATED_PAWN_MG * black_on_file
                eg -= ISOLATED_PAWN_EG * black_on_file

        white_passed_mask = PASSED_MASK[chess.WHITE]
        black_passed_mask = PASSED_MASK[chess.BLACK]
        for square in chess.scan_forward(white_pawns):
            if not (black_pawns & white_passed_mask[square]):
                rank_index = chess.square_rank(square)
                mg += PASSED_BONUS_MG[rank_index]
                eg += PASSED_BONUS_EG[rank_index]
        for square in chess.scan_forward(black_pawns):
            if not (white_pawns & black_passed_mask[square]):
                rank_index = 7 - chess.square_rank(square)
                mg -= PASSED_BONUS_MG[rank_index]
                eg -= PASSED_BONUS_EG[rank_index]

        if len(self.pawn_cache) > 200_000:
            self.pawn_cache.clear()
        result = (mg, eg)
        self.pawn_cache[cache_key] = result
        return result

    def evaluate(self, board: chess.Board) -> int:
        """Static evaluation from the side to move's point of view."""
        white_pawns = board.pawns & board.occupied_co[chess.WHITE]
        black_pawns = board.pawns & board.occupied_co[chess.BLACK]

        pawn_mg, pawn_eg = self.pawn_structure(white_pawns, black_pawns)
        mg = self.mg + pawn_mg
        eg = self.eg + pawn_eg

        # Bishop pair.
        if chess.popcount(board.bishops & board.occupied_co[chess.WHITE]) >= 2:
            mg += BISHOP_PAIR_MG
            eg += BISHOP_PAIR_EG
        if chess.popcount(board.bishops & board.occupied_co[chess.BLACK]) >= 2:
            mg -= BISHOP_PAIR_MG
            eg -= BISHOP_PAIR_EG

        # Rooks on open and semi-open files.
        all_pawns = white_pawns | black_pawns
        for square in chess.scan_forward(board.rooks & board.occupied_co[chess.WHITE]):
            file_mask = FILE_MASKS[chess.square_file(square)]
            if not (all_pawns & file_mask):
                mg += ROOK_OPEN_FILE
            elif not (white_pawns & file_mask):
                mg += ROOK_SEMI_OPEN_FILE
        for square in chess.scan_forward(board.rooks & board.occupied_co[chess.BLACK]):
            file_mask = FILE_MASKS[chess.square_file(square)]
            if not (all_pawns & file_mask):
                mg -= ROOK_OPEN_FILE
            elif not (black_pawns & file_mask):
                mg -= ROOK_SEMI_OPEN_FILE

        # King safety: pawns sheltering the king, minus enemy pieces nearby.
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is not None:
            zone = KING_ZONE[white_king]
            mg += KING_SHIELD_BONUS * chess.popcount(white_pawns & zone)
            attackers = chess.popcount(
                zone & board.occupied_co[chess.BLACK] & ~board.pawns
            )
            mg += KING_ATTACKER_PENALTY * attackers * attackers
        if black_king is not None:
            zone = KING_ZONE[black_king]
            mg -= KING_SHIELD_BONUS * chess.popcount(black_pawns & zone)
            attackers = chess.popcount(
                zone & board.occupied_co[chess.WHITE] & ~board.pawns
            )
            mg -= KING_ATTACKER_PENALTY * attackers * attackers

        phase = self.phase
        if phase > TOTAL_PHASE:
            phase = TOTAL_PHASE
        score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
        if board.turn == chess.BLACK:
            score = -score
        return score + TEMPO_BONUS

    # -- static exchange evaluation ----------------------------------------

    def see(self, board: chess.Board, move: chess.Move) -> int:
        """Material won or lost by the exchange sequence on `move.to_square`.

        A standard swap-off list. X-rays are handled by removing each attacker
        from the occupancy before asking python-chess for the next one, which
        lets a rook behind a rook join the exchange.
        """
        to_square = move.to_square
        from_square = move.from_square
        if board.is_en_passant(move):
            captured_value = SEE_VALUE[chess.PAWN]
        else:
            captured_piece = board.piece_type_at(to_square)
            captured_value = 0 if captured_piece is None else SEE_VALUE[captured_piece]

        attacker_type = board.piece_type_at(from_square)
        if attacker_type is None:  # pragma: no cover - defensive
            return 0
        if move.promotion:
            captured_value += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
            attacker_type = move.promotion

        gain = [0] * 32
        gain[0] = captured_value
        depth = 0
        occupied = board.occupied & ~chess.BB_SQUARES[from_square]
        if board.is_en_passant(move):
            captured_square = (
                to_square - 8 if board.turn == chess.WHITE else to_square + 8
            )
            occupied &= ~chess.BB_SQUARES[captured_square]
        side = not board.turn

        while True:
            attackers = board.attackers_mask(side, to_square, occupied) & occupied
            if not attackers:
                break
            # Pick the least valuable attacker available.
            chosen_square = -1
            chosen_type = 0
            for piece_type in range(1, 7):
                candidates = attackers & board.pieces_mask(piece_type, side)
                if candidates:
                    chosen_square = chess.lsb(candidates)
                    chosen_type = piece_type
                    break
            if chosen_square < 0:
                break
            depth += 1
            if depth >= 31:
                break
            gain[depth] = SEE_VALUE[attacker_type] - gain[depth - 1]
            attacker_type = chosen_type
            occupied &= ~chess.BB_SQUARES[chosen_square]
            side = not side
            # Once the exchange is hopeless for the side to move, stop.
            if max(-gain[depth - 1], gain[depth]) < 0:
                break

        while depth > 0:
            gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
            depth -= 1
        return gain[0]

    # -- move ordering -----------------------------------------------------

    def order_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        tt_move: Optional[chess.Move],
        ply: int,
        previous: Optional[chess.Move],
    ) -> List[chess.Move]:
        killer_one, killer_two = self.killers[ply]
        counter = None
        if previous is not None:
            counter = self.counter_moves.get(
                (previous.from_square, previous.to_square)
            )
        history = self.history
        turn = board.turn
        ep_square = board.ep_square
        scored: List[Tuple[int, chess.Move]] = []
        append = scored.append
        for move in moves:
            if move == tt_move:
                append((1 << 24, move))
                continue
            from_square = move.from_square
            to_square = move.to_square
            attacker = board.piece_type_at(from_square) or 1
            victim = board.piece_type_at(to_square)
            if victim is None and to_square == ep_square and attacker == chess.PAWN:
                victim = chess.PAWN
            if victim is not None or move.promotion:
                # Most Valuable Victim / Least Valuable Attacker is nearly free.
                # A full static exchange evaluation is only worth its cost when
                # MVV-LVA cannot tell whether the capture loses material.
                victim_value = SEE_VALUE[victim] if victim is not None else 0
                if move.promotion:
                    victim_value += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
                if victim_value >= SEE_VALUE[attacker]:
                    append(((1 << 22) + victim_value * 8 - attacker, move))
                elif self.see(board, move) >= 0:
                    append(((1 << 22) + victim_value * 8 - attacker, move))
                else:
                    append((-(1 << 22) + victim_value, move))
                continue
            if move == killer_one:
                append((1 << 21, move))
                continue
            if move == killer_two:
                append(((1 << 21) - 1, move))
                continue
            if counter is not None and move == counter:
                append((1 << 20, move))
                continue
            append((history[turn * 7 + attacker][to_square], move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [move for _, move in scored]

    # -- draw detection ----------------------------------------------------

    def is_draw(self, board: chess.Board, key: Hashable) -> bool:
        if board.halfmove_clock >= 100:
            return True
        if key in self.path_keys:
            return True
        if self.game_keys.get(key, 0) >= 1:
            return True
        return board.is_insufficient_material()

    # -- quiescence --------------------------------------------------------

    def quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int
    ) -> int:
        self.nodes += 1
        if not self.nodes & 1023:
            if time.monotonic() >= self.deadline:
                raise TimeUp
        if ply > self.seldepth:
            self.seldepth = ply
        if ply >= MAX_PLY - 1:
            return self.evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
            stand_pat = -INFINITY
        else:
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = [
                move
                for move in board.generate_legal_moves()
                if board.is_capture(move) or move.promotion == chess.QUEEN
            ]

        best = stand_pat
        ordered = self.order_moves(board, moves, None, ply, None)
        for move in ordered:
            if not in_check:
                # Delta pruning: skip captures that cannot lift alpha.
                captured = board.piece_type_at(move.to_square)
                gain = SEE_VALUE[captured] if captured is not None else SEE_VALUE[chess.PAWN]
                if move.promotion:
                    gain += SEE_VALUE[move.promotion]
                if stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
                if self.see(board, move) < 0:
                    continue
            self.push(board, move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            self.pop(board)
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- main search -------------------------------------------------------

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool,
        previous: Optional[chess.Move],
    ) -> int:
        self.nodes += 1
        if not self.nodes & 1023:
            if time.monotonic() >= self.deadline:
                raise TimeUp

        is_root = ply == 0
        is_pv = beta - alpha > 1

        key = POSITION_KEY(board)
        if not is_root and self.is_draw(board, key):
            return 0

        # Mate-distance pruning: never look for a mate longer than one found.
        if not is_root:
            alpha = max(alpha, -MATE_SCORE + ply)
            beta = min(beta, MATE_SCORE - ply - 1)
            if alpha >= beta:
                return alpha

        tt_move: Optional[chess.Move] = None
        entry = self.tt.get(key)
        if entry is not None:
            entry_depth: int = entry[0]
            entry_score: int = entry[1]
            entry_flag: int = entry[2]
            entry_move = entry[3]
            tt_move = entry_move
            if entry_depth >= depth and not is_pv:
                if entry_score > MATE_THRESHOLD:
                    entry_score -= ply
                elif entry_score < -MATE_THRESHOLD:
                    entry_score += ply
                if entry_flag == TT_EXACT:
                    return entry_score
                if entry_flag == TT_LOWER and entry_score >= beta:
                    return entry_score
                if entry_flag == TT_UPPER and entry_score <= alpha:
                    return entry_score

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        if ply >= MAX_PLY - 1:
            return self.evaluate(board)

        static_eval = -INFINITY if in_check else self.evaluate(board)

        # Side to move has pieces other than pawns? Needed to keep null-move
        # and reduction heuristics away from zugzwang-prone endings.
        own = board.occupied_co[board.turn]
        has_pieces = bool(own & ~(board.pawns | board.kings))

        if not is_pv and not in_check:
            # Reverse futility: the position is so good that even giving up
            # a margin per remaining ply leaves us above beta.
            if depth <= 6 and static_eval - REVERSE_FUTILITY_MARGIN * depth >= beta:
                return static_eval

            # Null-move pruning.
            if (
                can_null
                and depth >= 3
                and has_pieces
                and static_eval >= beta
            ):
                reduction = NULL_MOVE_BASE_REDUCTION + depth // 6
                board.push(chess.Move.null())
                self.mg_stack.append((self.mg, self.eg, self.phase))
                self.path_keys.append(key)
                try:
                    score = -self.negamax(
                        board,
                        depth - reduction - 1,
                        -beta,
                        -beta + 1,
                        ply + 1,
                        False,
                        None,
                    )
                finally:
                    self.path_keys.pop()
                    board.pop()
                    self.mg, self.eg, self.phase = self.mg_stack.pop()
                if score >= beta:
                    if score > MATE_THRESHOLD:
                        score = beta
                    return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        # Internal iterative deepening: without a hash move, ordering at a PV
        # node is poor enough to be worth a shallow search to find one.
        if tt_move is None and is_pv and depth >= 5:
            self.negamax(board, depth - 2, alpha, beta, ply, False, previous)
            entry = self.tt.get(key)
            if entry is not None:
                tt_move = entry[3]

        ordered = self.order_moves(board, moves, tt_move, ply, previous)

        best_score = -INFINITY
        best_move: Optional[chess.Move] = None
        original_alpha = alpha
        self.path_keys.append(key)
        history = self.history
        turn = board.turn
        move_count = 0
        futile = (
            not is_pv
            and not in_check
            and depth <= 3
            and static_eval + FUTILITY_MARGIN * depth <= alpha
        )

        try:
            for move in ordered:
                move_count += 1
                captured = board.piece_type_at(move.to_square)
                is_quiet = (
                    captured is None
                    and not move.promotion
                    and not board.is_en_passant(move)
                )

                if is_quiet and best_move is not None:
                    # Futility pruning: a quiet move here cannot reach alpha.
                    if futile:
                        continue
                    # Late move pruning: deep in a bad move list, stop looking.
                    if (
                        not is_pv
                        and depth <= 4
                        and move_count > 6 + depth * depth
                        and has_pieces
                    ):
                        continue

                self.push(board, move)
                gives_check = board.is_check()

                # Late move reductions.
                reduction = 0
                if (
                    depth >= 3
                    and move_count > 3
                    and is_quiet
                    and not in_check
                    and not gives_check
                ):
                    reduction = 1 + (depth >= 6) + (move_count > 8)
                    if is_pv:
                        reduction -= 1
                    if reduction < 0:
                        reduction = 0
                    if reduction > depth - 2:
                        reduction = depth - 2

                if move_count == 1:
                    score = -self.negamax(
                        board, depth - 1, -beta, -alpha, ply + 1, True, move
                    )
                else:
                    score = -self.negamax(
                        board,
                        depth - 1 - reduction,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        True,
                        move,
                    )
                    if score > alpha and reduction:
                        score = -self.negamax(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1, True, move
                        )
                    if alpha < score < beta:
                        score = -self.negamax(
                            board, depth - 1, -beta, -alpha, ply + 1, True, move
                        )
                self.pop(board)

                if score > best_score:
                    best_score = score
                    best_move = move
                    if is_root:
                        self.root_best = move
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if is_quiet:
                                killers = self.killers[ply]
                                if killers[0] != move:
                                    killers[1] = killers[0]
                                    killers[0] = move
                                piece_type = board.piece_type_at(move.from_square) or 1
                                bucket = history[turn * 7 + piece_type]
                                bucket[move.to_square] += depth * depth
                                if bucket[move.to_square] > 1 << 18:
                                    for index in range(64):
                                        bucket[index] >>= 1
                                if previous is not None:
                                    self.counter_moves[
                                        (previous.from_square, previous.to_square)
                                    ] = move
                            break
        finally:
            self.path_keys.pop()

        if best_score >= beta:
            flag = TT_LOWER
        elif best_score > original_alpha:
            flag = TT_EXACT
        else:
            flag = TT_UPPER
        stored = best_score
        if stored > MATE_THRESHOLD:
            stored += ply
        elif stored < -MATE_THRESHOLD:
            stored -= ply
        if len(self.tt) < MAX_TT_ENTRIES:
            self.tt[key] = (depth, stored, flag, best_move)
        else:
            existing = self.tt.get(key)
            if existing is None or existing[0] <= depth:
                self.tt[key] = (depth, stored, flag, best_move)
        return best_score

    # -- iterative deepening ----------------------------------------------

    def search(self, board: chess.Board, budget_seconds: float) -> chess.Move:
        self.set_position(board)
        self.nodes = 0
        self.seldepth = 0
        self.deadline = time.monotonic() + budget_seconds
        self.path_keys.clear()
        for killers in self.killers:
            killers[0] = None
            killers[1] = None

        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves")
        self.root_best = legal[0]
        best_move = legal[0]
        if len(legal) == 1:
            return best_move

        score = 0
        start = time.monotonic()
        # Aborting the search throws out of the middle of the recursion, which
        # leaves moves pushed on the board. Remember where the real position
        # sits so we can unwind back to it rather than returning a move that
        # is legal only in some position deep inside the tree.
        base_ply = len(board.move_stack)
        base_scores = (self.mg, self.eg, self.phase)

        for depth in range(1, MAX_PLY):
            self.root_best = best_move
            try:
                if depth <= 3:
                    score = self.negamax(
                        board, depth, -INFINITY, INFINITY, 0, True, None
                    )
                else:
                    # Aspiration window around the previous iteration's score.
                    delta = ASPIRATION_DELTA
                    alpha = score - delta
                    beta = score + delta
                    while True:
                        score = self.negamax(board, depth, alpha, beta, 0, True, None)
                        if score <= alpha:
                            beta = (alpha + beta) // 2
                            alpha = score - delta
                            delta += delta // 2
                        elif score >= beta:
                            beta = score + delta
                            delta += delta // 2
                        else:
                            break
            except TimeUp:
                while len(board.move_stack) > base_ply:
                    board.pop()
                self.mg_stack.clear()
                self.path_keys.clear()
                self.mg, self.eg, self.phase = base_scores
                if self.root_best is not None:
                    best_move = self.root_best
                break

            if self.root_best is not None:
                best_move = self.root_best

            elapsed = time.monotonic() - start
            if not _QUIET:
                print(
                    f"depth {depth} seldepth {self.seldepth} score {score} "
                    f"nodes {self.nodes} time {elapsed:.2f}s pv {best_move.uci()}"
                )
            if abs(score) > MATE_THRESHOLD:
                break
            # If the next iteration plainly cannot finish, stop now and keep
            # the clock rather than throwing the work away.
            if elapsed > budget_seconds * 0.55:
                break
        return best_move


# --------------------------------------------------------------------------
# Module-level state and the competition entry point
# --------------------------------------------------------------------------

_SEARCHER = Searcher()

# Set while the warm-up search runs at import so it does not fill the first
# 4 KB of the log the platform keeps, which is where the init timings go.
_QUIET = False

# Wall-clock overhead reserved for serialising the move and the runner's own
# bookkeeping. The referee is unforgiving, so this is deliberately generous.
SAFETY_MARGIN_MS = 120
INCREMENT_MS = 500


def _budget_seconds(board: chess.Board, time_left_ms: int) -> float:
    """Decide how long to think, from the clock we were actually handed."""
    remaining = max(0, time_left_ms - SAFETY_MARGIN_MS)
    if remaining <= 0:
        return 0.005

    # Rough guess at how many moves are left to play, from how much material
    # is still on. Fewer pieces means fewer moves to budget for.
    pieces = chess.popcount(board.occupied)
    expected_moves = 20 if pieces > 20 else (16 if pieces > 10 else 12)

    # The increment lands after this move, so it is safe to spend it.
    budget = remaining / expected_moves + INCREMENT_MS * 0.75

    # Never risk more than a third of what is left in one move, and never
    # bother thinking for longer than a sane cap.
    hard_cap = remaining * 0.33
    if budget > hard_cap:
        budget = hard_cap
    if remaining < 3000:
        # Emergency: move fast and keep the increment ahead of the clock.
        budget = min(budget, remaining * 0.20)
    return max(0.005, budget / 1000.0)


def get_move(fen: str, time_left_ms: int) -> str:
    """Competition entry point. Returns a legal UCI move for `fen`."""
    board = chess.Board(fen)

    try:
        # Track the positions we have been asked about so the search knows
        # which repetitions are already on the board. Counting starts at the
        # first FEN of the game, which is where the referee counts from too:
        # rated games begin from a curated opening, not the standard start.
        key = POSITION_KEY(board)
        _SEARCHER.game_keys[key] = _SEARCHER.game_keys.get(key, 0) + 1

        budget = _budget_seconds(board, time_left_ms)
        move = _SEARCHER.search(board, budget)
        # Always check legality against a clean board built from the FEN we
        # were given, never against the board the search has been mutating.
        verifier = chess.Board(fen)
        if move not in verifier.legal_moves:  # pragma: no cover - defensive
            print(f"search returned {move.uci()}, illegal here; falling back")
            move = next(iter(verifier.legal_moves))
        return move.uci()
    except Exception as exc:  # pragma: no cover - never lose on a crash
        print(f"fallback after {type(exc).__name__}: {exc}")
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return "0000"
        # Prefer a capture over a random move if we have to bail out.
        best = legal[0]
        best_value = -1
        for candidate in legal:
            victim = board.piece_type_at(candidate.to_square)
            value = SEE_VALUE[victim] if victim is not None else 0
            if value > best_value:
                best_value = value
                best = candidate
        return best.uci()


def _warm_up() -> None:
    """Touch the hot paths once during the 90s init budget.

    Anything deferred to the first `get_move` comes out of the match clock
    instead, so the piece-square tables, the evaluation and the search are all
    exercised here on a throwaway searcher.
    """
    global _QUIET
    _QUIET = True
    try:
        Searcher().search(chess.Board(), 0.35)
    except Exception:  # pragma: no cover - warm-up must never break import
        pass
    finally:
        _QUIET = False


_warm_up()
print("agent ready")
