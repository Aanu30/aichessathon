"""Search and evaluation, jitted with numba, on top of fastcore's move generation.

The entire search runs inside one jitted call per iteration of iterative
deepening. Crossing the Python boundary costs about 1.5 microseconds, which is
worth roughly a hundred nodes here, so the boundary is crossed once per depth
rather than once per node.

Time is handled by a node budget rather than a clock check, because numba
cannot read the clock without an expensive object-mode escape. Python measures
the achieved node rate, converts the remaining milliseconds into a node cap,
and the search aborts cleanly when it reaches it. Iterative deepening means an
abort always leaves a completed shallower result to return.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fastcore import (
    BOCC,
    K,
    P,
    Q,
    R,
    U64,
    WHITE,
    WOCC,
    attacked_by,
    generate_moves,
    lsb,
    make_move,
    piece_at,
    popcount,
    unmake_move,
)

ONE = U64(1)

MATE = 30000
MATE_THRESHOLD = 29000
INF = 32000
MAX_PLY = 64

TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = U64(TT_SIZE - 1)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# Indices into the `out` array returned to Python.
OUT_SCORE, OUT_MOVE, OUT_NODES, OUT_ABORT, OUT_SELDEPTH = 0, 1, 2, 3, 4

# Midgame and endgame material, indexed by piece. PeSTO's constants.
MG_VAL = np.array([82, 337, 365, 477, 1025, 0], dtype=np.int64)
EG_VAL = np.array([94, 281, 297, 512, 936, 0], dtype=np.int64)
PHASE_W = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
SEE_VAL = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)

_MG_TABLES = [
    # pawn
    [0,0,0,0,0,0,0,0, 98,134,61,95,68,126,34,-11, -6,7,26,31,65,56,25,-20,
     -14,13,6,21,23,12,17,-23, -27,-2,-5,12,17,6,10,-25, -26,-4,-4,-10,3,3,33,-12,
     -35,-1,-20,-23,-15,24,38,-22, 0,0,0,0,0,0,0,0],
    # knight
    [-167,-89,-34,-49,61,-97,-15,-107, -73,-41,72,36,23,62,7,-17,
     -47,60,37,65,84,129,73,44, -9,17,19,53,37,69,18,22, -13,4,16,13,28,19,21,-8,
     -23,-9,12,10,19,17,25,-16, -29,-53,-12,-3,-1,18,-14,-19, -105,-21,-58,-33,-17,-28,-19,-23],
    # bishop
    [-29,4,-82,-37,-25,-42,7,-8, -26,16,-18,-13,30,59,18,-47, -16,37,43,40,35,50,37,-2,
     -4,5,19,50,37,37,7,-2, -6,13,13,26,34,12,10,4, 0,15,15,15,14,27,18,10,
     4,15,16,0,7,21,33,1, -33,-3,-14,-21,-13,-12,-39,-21],
    # rook
    [32,42,32,51,63,9,31,43, 27,32,58,62,80,67,26,44, -5,19,26,36,17,45,61,16,
     -24,-11,7,26,24,35,-8,-20, -36,-26,-12,-1,9,-7,6,-23, -45,-25,-16,-17,3,0,-5,-33,
     -44,-16,-20,-9,-1,11,-6,-71, -19,-13,1,17,16,7,-37,-26],
    # queen
    [-28,0,29,12,59,44,43,45, -24,-39,-5,1,-16,57,28,54, -13,-17,7,8,29,56,47,57,
     -27,-27,-16,-16,-1,17,-2,1, -9,-26,-9,-10,-2,-4,3,-3, -14,2,-11,-2,-5,2,14,5,
     -35,-8,11,2,8,15,-3,1, -1,-18,-9,10,-15,-25,-31,-50],
    # king
    [-65,23,16,-15,-56,-34,2,13, 29,-1,-20,-7,-8,-4,-38,-29, -9,24,2,-16,-20,6,22,-22,
     -17,-20,-12,-27,-30,-25,-14,-36, -49,-1,-27,-39,-46,-44,-33,-51,
     -14,-14,-22,-46,-44,-30,-15,-27, 1,7,-8,-64,-43,-16,9,8, -15,36,12,-54,8,-28,24,14],
]
_EG_TABLES = [
    [0,0,0,0,0,0,0,0, 178,173,158,134,147,132,165,187, 94,100,85,67,56,53,82,84,
     32,24,13,5,-2,4,17,17, 13,9,-3,-7,-7,-8,3,-1, 4,7,-6,1,0,-5,-1,-8,
     13,8,8,10,13,0,2,-7, 0,0,0,0,0,0,0,0],
    [-58,-38,-13,-28,-31,-27,-63,-99, -25,-8,-25,-2,-9,-25,-24,-52, -24,-20,10,9,-1,-9,-19,-41,
     -17,3,22,22,22,11,8,-18, -18,-6,16,25,16,17,4,-18, -23,-3,-1,15,10,-3,-20,-22,
     -42,-20,-10,-5,-2,-20,-23,-44, -29,-51,-23,-15,-22,-18,-50,-64],
    [-14,-21,-11,-8,-7,-9,-17,-24, -8,-4,7,-12,-3,-13,-4,-14, 2,-8,0,-1,-2,6,0,4,
     -3,9,12,9,14,10,3,2, -6,3,13,19,7,10,-3,-9, -12,-3,8,10,13,3,-7,-15,
     -14,-18,-7,-1,4,-9,-15,-27, -23,-9,-23,-5,-9,-16,-5,-17],
    [13,10,18,15,12,12,8,5, 11,13,13,11,-3,3,8,3, 7,7,7,5,4,-3,-5,-3,
     4,3,13,1,2,1,-1,2, 3,5,8,4,-5,-6,-8,-11, -4,0,-5,-1,-7,-12,-8,-16,
     -6,-6,0,2,-9,-9,-11,-3, -9,2,3,-1,-5,-13,4,-20],
    [-9,22,22,27,27,19,10,20, -17,20,32,41,58,25,30,0, -20,6,9,49,47,35,19,9,
     3,22,24,45,57,40,57,36, -18,28,19,47,31,34,39,23, -16,-27,15,6,9,17,10,5,
     -22,-23,-30,-16,-16,-23,-36,-32, -33,-28,-22,-43,-5,-32,-20,-41],
    [-74,-35,-18,-18,-11,15,4,-17, -12,17,14,17,17,38,23,11, 10,17,23,15,20,45,44,13,
     -8,22,24,27,26,33,26,3, -18,-4,21,24,27,23,9,-11, -19,-3,11,21,23,16,7,-9,
     -27,-11,4,13,14,4,-5,-17, -53,-34,-21,-11,-28,-14,-24,-43],
]


def _build_pst() -> tuple[np.ndarray, np.ndarray]:
    """PST[colour * 6 + piece][square], material folded in.

    Table row 0 is rank 8, so white reads `square ^ 56` and black reads
    `square` directly. Precomputing both removes a branch from the hot loop.
    """
    mg = np.zeros((12, 64), dtype=np.int64)
    eg = np.zeros((12, 64), dtype=np.int64)
    for piece in range(6):
        for square in range(64):
            mg[piece][square] = _MG_TABLES[piece][square ^ 56] + MG_VAL[piece]
            eg[piece][square] = _EG_TABLES[piece][square ^ 56] + EG_VAL[piece]
            mg[6 + piece][square] = _MG_TABLES[piece][square] + MG_VAL[piece]
            eg[6 + piece][square] = _EG_TABLES[piece][square] + EG_VAL[piece]
    return mg, eg


PST_MG, PST_EG = _build_pst()

_rng = np.random.RandomState(20260907)
ZOB_PIECE = _rng.randint(0, 1 << 63, size=(12, 64), dtype=np.int64).astype(np.uint64)
ZOB_SIDE = np.uint64(_rng.randint(0, 1 << 63))
ZOB_CASTLE = _rng.randint(0, 1 << 63, size=16, dtype=np.int64).astype(np.uint64)
ZOB_EP = _rng.randint(0, 1 << 63, size=64, dtype=np.int64).astype(np.uint64)


@njit(cache=False)
def zobrist(bb: np.ndarray, st: np.ndarray) -> np.uint64:
    """Full hash. Only called at the root; the search updates it by XOR."""
    h = U64(0)
    for piece in range(6):
        white = bb[piece] & bb[WOCC]
        while white:
            square = lsb(white)
            white &= white - ONE
            h ^= ZOB_PIECE[piece][square]
        black = bb[piece] & bb[BOCC]
        while black:
            square = lsb(black)
            black &= black - ONE
            h ^= ZOB_PIECE[6 + piece][square]
    h ^= ZOB_CASTLE[st[1]]
    if st[2] >= 0:
        h ^= ZOB_EP[st[2]]
    if st[0] != WHITE:
        h ^= ZOB_SIDE
    return h


@njit(cache=False)
def evaluate(bb: np.ndarray, st: np.ndarray) -> np.int64:
    """Tapered material and piece-square score, from the mover's point of view."""
    mg = 0
    eg = 0
    phase = 0
    for piece in range(6):
        white = bb[piece] & bb[WOCC]
        while white:
            square = lsb(white)
            white &= white - ONE
            mg += PST_MG[piece][square]
            eg += PST_EG[piece][square]
            phase += PHASE_W[piece]
        black = bb[piece] & bb[BOCC]
        while black:
            square = lsb(black)
            black &= black - ONE
            mg -= PST_MG[6 + piece][square]
            eg -= PST_EG[6 + piece][square]
            phase += PHASE_W[piece]
    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24
    if st[0] != WHITE:
        score = -score
    return score + 12


@njit(cache=False)
def see_capture(bb: np.ndarray, st: np.ndarray, move: np.int64) -> np.int64:
    """Cheap static exchange estimate: victim value minus attacker if defended."""
    to = (move >> 6) & 63
    frm = move & 63
    victim = piece_at(bb, to)
    attacker = piece_at(bb, frm)
    if victim < 0:
        return 0
    gain = SEE_VAL[victim]
    enemy = 1 - st[0]
    if attacked_by(bb, to, enemy):
        gain -= SEE_VAL[attacker]
    return gain


@njit(cache=False)
def score_moves(
    bb: np.ndarray,
    st: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    count: np.int64,
    tt_move: np.int64,
    killers: np.ndarray,
    history: np.ndarray,
    ply: np.int64,
) -> None:
    """Order: hash move, then captures by MVV-LVA, then killers, then history."""
    for index in range(count):
        move = moves[index]
        if move == tt_move:
            scores[index] = 1 << 30
            continue
        to = (move >> 6) & 63
        frm = move & 63
        promo = (move >> 12) & 7
        victim = piece_at(bb, to)
        if victim >= 0 or promo:
            attacker = piece_at(bb, frm)
            value = SEE_VAL[victim] if victim >= 0 else 0
            if promo:
                value += SEE_VAL[promo]
            scores[index] = (1 << 28) + value * 16 - (attacker if attacker >= 0 else 0)
        elif move == killers[ply, 0]:
            scores[index] = 1 << 27
        elif move == killers[ply, 1]:
            scores[index] = (1 << 27) - 1
        else:
            piece = piece_at(bb, frm)
            scores[index] = history[st[0] * 6 + (piece if piece >= 0 else 0), to]


@njit(cache=False)
def pick_move(scores: np.ndarray, moves: np.ndarray, count: np.int64, index: np.int64) -> np.int64:
    """Selection sort, one step. Cheaper than sorting a list we may abandon."""
    best = index
    for other in range(index + 1, count):
        if scores[other] > scores[best]:
            best = other
    if best != index:
        scores[index], scores[best] = scores[best], scores[index]
        moves[index], moves[best] = moves[best], moves[index]
    return moves[index]


@njit(cache=False)
def in_check(bb: np.ndarray, st: np.ndarray) -> bool:
    king = bb[K] & (bb[WOCC] if st[0] == WHITE else bb[BOCC])
    if not king:
        return False
    return attacked_by(bb, lsb(king), 1 - st[0])


@njit(cache=False)
def quiescence(
    bb: np.ndarray,
    st: np.ndarray,
    alpha: np.int64,
    beta: np.int64,
    ply: np.int64,
    moves: np.ndarray,
    scores: np.ndarray,
    undo: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    out: np.ndarray,
) -> np.int64:
    out[OUT_NODES] += 1
    if ply > out[OUT_SELDEPTH]:
        out[OUT_SELDEPTH] = ply
    if ply >= MAX_PLY - 2:
        return evaluate(bb, st)

    stand = evaluate(bb, st)
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand

    count = generate_moves(bb, st, moves[ply])
    score_moves(bb, st, moves[ply], scores[ply], count, -1, killers, history, ply)

    best = stand
    for index in range(count):
        move = pick_move(scores[ply], moves[ply], count, index)
        to = (move >> 6) & 63
        promo = (move >> 12) & 7
        victim = piece_at(bb, to)
        flag = (move >> 15) & 7
        # Captures and queen promotions only.
        if victim < 0 and promo != 4 and flag != 1:
            continue
        # Delta pruning, and skip captures that lose material outright.
        if victim >= 0 and stand + SEE_VAL[victim] + 200 < alpha:
            continue
        if victim >= 0 and see_capture(bb, st, move) < 0:
            continue
        if not make_move(bb, st, move, undo, ply):
            continue
        value = -quiescence(
            bb, st, -beta, -alpha, ply + 1, moves, scores, undo, killers, history, out
        )
        unmake_move(bb, st, move, undo, ply)
        if value > best:
            best = value
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    break
    return best


@njit(cache=False)
def negamax(
    bb: np.ndarray,
    st: np.ndarray,
    depth: np.int64,
    alpha: np.int64,
    beta: np.int64,
    ply: np.int64,
    can_null: np.int64,
    moves: np.ndarray,
    scores: np.ndarray,
    undo: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    tt_key: np.ndarray,
    tt_move: np.ndarray,
    tt_score: np.ndarray,
    tt_depth: np.ndarray,
    tt_flag: np.ndarray,
    hist: np.ndarray,
    hist_len: np.int64,
    node_cap: np.int64,
    out: np.ndarray,
) -> np.int64:
    out[OUT_NODES] += 1
    if out[OUT_NODES] >= node_cap:
        out[OUT_ABORT] = 1
        return 0

    is_root = ply == 0
    is_pv = beta - alpha > 1
    key = zobrist(bb, st)

    if not is_root:
        # Repetition or fifty-move: a draw.
        if st[3] >= 100:
            return 0
        repeats = 0
        for index in range(hist_len):
            if hist[index] == key:
                repeats += 1
        if repeats >= 1:
            return 0

    slot = np.int64(key & TT_MASK)
    hash_move = np.int64(-1)
    if tt_key[slot] == key:
        hash_move = tt_move[slot]
        if not is_pv and tt_depth[slot] >= depth:
            stored = tt_score[slot]
            if stored > MATE_THRESHOLD:
                stored -= ply
            elif stored < -MATE_THRESHOLD:
                stored += ply
            flag = tt_flag[slot]
            if flag == TT_EXACT:
                return stored
            if flag == TT_LOWER and stored >= beta:
                return stored
            if flag == TT_UPPER and stored <= alpha:
                return stored

    checked = in_check(bb, st)
    if checked:
        depth += 1

    if depth <= 0:
        return quiescence(
            bb, st, alpha, beta, ply, moves, scores, undo, killers, history, out
        )

    static = evaluate(bb, st)
    own = bb[WOCC] if st[0] == WHITE else bb[BOCC]
    has_pieces = (own & ~(bb[P] | bb[K])) != U64(0)

    if not is_pv and not checked:
        if depth <= 6 and static - 85 * depth >= beta:
            return static
        if can_null and depth >= 3 and has_pieces and static >= beta:
            saved_ep = st[2]
            st[0] = 1 - st[0]
            st[2] = -1
            hist[hist_len] = key
            reduction = 3 + depth // 6
            value = -negamax(
                bb, st, depth - reduction - 1, -beta, -beta + 1, ply + 1, 0,
                moves, scores, undo, killers, history,
                tt_key, tt_move, tt_score, tt_depth, tt_flag,
                hist, hist_len + 1, node_cap, out,
            )
            st[0] = 1 - st[0]
            st[2] = saved_ep
            if out[OUT_ABORT]:
                return 0
            if value >= beta:
                return beta if value > MATE_THRESHOLD else value

    count = generate_moves(bb, st, moves[ply])
    score_moves(bb, st, moves[ply], scores[ply], count, hash_move, killers, history, ply)

    hist[hist_len] = key
    best_score = -INF
    best_move = np.int64(-1)
    original_alpha = alpha
    legal = 0

    for index in range(count):
        move = pick_move(scores[ply], moves[ply], count, index)
        to = (move >> 6) & 63
        victim = piece_at(bb, to)
        promo = (move >> 12) & 7
        quiet = victim < 0 and promo == 0

        if not make_move(bb, st, move, undo, ply):
            continue
        legal += 1

        reduction = 0
        if depth >= 3 and legal > 3 and quiet and not checked:
            reduction = 1 + (1 if depth >= 6 else 0) + (1 if legal > 8 else 0)
            if is_pv:
                reduction -= 1
            if reduction < 0:
                reduction = 0
            if reduction > depth - 2:
                reduction = depth - 2

        if legal == 1:
            value = -negamax(
                bb, st, depth - 1, -beta, -alpha, ply + 1, 1,
                moves, scores, undo, killers, history,
                tt_key, tt_move, tt_score, tt_depth, tt_flag,
                hist, hist_len + 1, node_cap, out,
            )
        else:
            value = -negamax(
                bb, st, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, 1,
                moves, scores, undo, killers, history,
                tt_key, tt_move, tt_score, tt_depth, tt_flag,
                hist, hist_len + 1, node_cap, out,
            )
            if value > alpha and reduction:
                value = -negamax(
                    bb, st, depth - 1, -alpha - 1, -alpha, ply + 1, 1,
                    moves, scores, undo, killers, history,
                    tt_key, tt_move, tt_score, tt_depth, tt_flag,
                    hist, hist_len + 1, node_cap, out,
                )
            if alpha < value < beta:
                value = -negamax(
                    bb, st, depth - 1, -beta, -alpha, ply + 1, 1,
                    moves, scores, undo, killers, history,
                    tt_key, tt_move, tt_score, tt_depth, tt_flag,
                    hist, hist_len + 1, node_cap, out,
                )
        unmake_move(bb, st, move, undo, ply)

        if out[OUT_ABORT]:
            return 0

        if value > best_score:
            best_score = value
            best_move = move
            if is_root:
                out[OUT_MOVE] = move
            if value > alpha:
                alpha = value
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != move:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        piece = piece_at(bb, move & 63)
                        if piece >= 0:
                            history[st[0] * 6 + piece, to] += depth * depth
                    break

    if legal == 0:
        return -MATE + ply if checked else 0

    stored = best_score
    if stored > MATE_THRESHOLD:
        stored += ply
    elif stored < -MATE_THRESHOLD:
        stored -= ply
    if tt_key[slot] != key or tt_depth[slot] <= depth:
        tt_key[slot] = key
        tt_move[slot] = best_move
        tt_score[slot] = stored
        tt_depth[slot] = depth
        if best_score >= beta:
            tt_flag[slot] = TT_LOWER
        elif best_score > original_alpha:
            tt_flag[slot] = TT_EXACT
        else:
            tt_flag[slot] = TT_UPPER
    return best_score


@njit(cache=False)
def search_depth(
    bb: np.ndarray,
    st: np.ndarray,
    depth: np.int64,
    alpha: np.int64,
    beta: np.int64,
    moves: np.ndarray,
    scores: np.ndarray,
    undo: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    tt_key: np.ndarray,
    tt_move: np.ndarray,
    tt_score: np.ndarray,
    tt_depth: np.ndarray,
    tt_flag: np.ndarray,
    hist: np.ndarray,
    hist_len: np.int64,
    node_cap: np.int64,
    out: np.ndarray,
) -> None:
    """One iteration of iterative deepening. Python drives the loop and clock."""
    out[OUT_ABORT] = 0
    out[OUT_SELDEPTH] = 0
    value = negamax(
        bb, st, depth, alpha, beta, 0, 1,
        moves, scores, undo, killers, history,
        tt_key, tt_move, tt_score, tt_depth, tt_flag,
        hist, hist_len, node_cap, out,
    )
    out[OUT_SCORE] = value
