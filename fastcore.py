"""Bitboard move generation, jitted with numba.

Note on `inline`: numba can inline at the IR level, which is 10% faster here
but costs 25 extra seconds of compilation out of a 90 second init budget.
That trade is not available to us, so the hints are deliberately absent.

Written because python-chess caps the search at roughly 25,000 nodes per
second, which is depth 9 or 10 in the middlegame. Everything here is plain
scalars and numpy arrays so numba can compile it, and the whole search will
eventually live inside this boundary: a Python-to-jit call costs about 1.5
microseconds, which is a hundred nodes' worth of work, so crossing per node
would waste the entire gain.

Squares are 0 = a1 to 63 = h8, matching python-chess, so perft and move lists
can be cross-checked against it directly.

State layout
------------
`bb` is a uint64[8]: six piece bitboards then the two colour occupancies.
`st` is an int64[5]: side to move, castling rights, en-passant square,
halfmove clock, and the king-capture flag used to reject illegal moves.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Piece indices into `bb`.
P, N, B, R, Q, K = 0, 1, 2, 3, 4, 5
WOCC, BOCC = 6, 7

# Indices into `st`.
STM, CASTLE, EP, HALF = 0, 1, 2, 3

WHITE, BLACK = 0, 1

U64 = np.uint64
ONE = U64(1)
FULL = U64(0xFFFFFFFFFFFFFFFF)

# Castling right bits.
WK, WQ, BK, BQ = 1, 2, 4, 8

# Move encoding: from | to << 6 | promo << 12 | flag << 15
# promo: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen
# flag:  0 normal, 1 en passant, 2 castling, 3 double pawn push
FLAG_EP, FLAG_CASTLE, FLAG_DOUBLE = 1, 2, 3


def _build_tables() -> tuple[np.ndarray, ...]:
    """Precompute attack tables. Pure Python: this runs once at import."""
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    pawn_att = np.zeros((2, 64), dtype=np.uint64)
    rays = np.zeros((8, 64), dtype=np.uint64)

    # Direction order: 0 N, 1 E, 2 S, 3 W, 4 NE, 5 SE, 6 SW, 7 NW.
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    for square in range(64):
        file_index, rank_index = square % 8, square // 8

        for df, dr in [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2),
        ]:
            f, r = file_index + df, rank_index + dr
            if 0 <= f < 8 and 0 <= r < 8:
                knight[square] |= np.uint64(1) << np.uint64(r * 8 + f)

        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = file_index + df, rank_index + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king[square] |= np.uint64(1) << np.uint64(r * 8 + f)

        for df in (-1, 1):
            f, r = file_index + df, rank_index + 1
            if 0 <= f < 8 and 0 <= r < 8:
                pawn_att[WHITE][square] |= np.uint64(1) << np.uint64(r * 8 + f)
            f, r = file_index + df, rank_index - 1
            if 0 <= f < 8 and 0 <= r < 8:
                pawn_att[BLACK][square] |= np.uint64(1) << np.uint64(r * 8 + f)

        for direction, (df, dr) in enumerate(deltas):
            f, r = file_index + df, rank_index + dr
            while 0 <= f < 8 and 0 <= r < 8:
                rays[direction][square] |= np.uint64(1) << np.uint64(r * 8 + f)
                f += df
                r += dr

    return knight, king, pawn_att, rays


KNIGHT_ATT, KING_ATT, PAWN_ATT, RAYS = _build_tables()

# Directions whose blocker is the least significant bit of the intersection.
POSITIVE_DIRS = np.array([0, 1, 4, 7], dtype=np.int64)
NEGATIVE_DIRS = np.array([2, 3, 5, 6], dtype=np.int64)
ROOK_DIRS = np.array([0, 1, 2, 3], dtype=np.int64)
BISHOP_DIRS = np.array([4, 5, 6, 7], dtype=np.int64)
IS_POSITIVE = np.array([1, 1, 0, 0, 1, 0, 0, 1], dtype=np.int64)


@njit(cache=False)
def msb(b: np.uint64) -> np.int64:
    """Index of the most significant set bit. Binary search, not a loop."""
    r = 0
    if b >> U64(32):
        b >>= U64(32)
        r += 32
    if b >> U64(16):
        b >>= U64(16)
        r += 16
    if b >> U64(8):
        b >>= U64(8)
        r += 8
    if b >> U64(4):
        b >>= U64(4)
        r += 4
    if b >> U64(2):
        b >>= U64(2)
        r += 2
    if b >> U64(1):
        r += 1
    return r


@njit(cache=False)
def lsb(b: np.uint64) -> np.int64:
    return msb(b & (~b + ONE))


@njit(cache=False)
def popcount(b: np.uint64) -> np.int64:
    c = 0
    while b:
        b &= b - ONE
        c += 1
    return c


@njit(cache=False)
def ray_attacks(square: np.int64, occupied: np.uint64, direction: np.int64) -> np.uint64:
    """Sliding attacks along one ray, stopping at the first blocker."""
    attacks = RAYS[direction][square]
    blockers = attacks & occupied
    if blockers:
        if IS_POSITIVE[direction]:
            first = lsb(blockers)
        else:
            first = msb(blockers)
        attacks &= ~RAYS[direction][first]
    return attacks


@njit(cache=False)
def rook_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    return (
        ray_attacks(square, occupied, 0)
        | ray_attacks(square, occupied, 1)
        | ray_attacks(square, occupied, 2)
        | ray_attacks(square, occupied, 3)
    )


@njit(cache=False)
def bishop_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    return (
        ray_attacks(square, occupied, 4)
        | ray_attacks(square, occupied, 5)
        | ray_attacks(square, occupied, 6)
        | ray_attacks(square, occupied, 7)
    )


@njit(cache=False)
def queen_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


@njit(cache=False)
def attacked_by(bb: np.ndarray, square: np.int64, by_colour: np.int64) -> bool:
    """Is `square` attacked by `by_colour`?"""
    occupied = bb[WOCC] | bb[BOCC]
    own = bb[WOCC] if by_colour == WHITE else bb[BOCC]
    # Pawns: a pawn attacks `square` iff it sits on a square that our own
    # pawn-attack table from `square` (for the opposite colour) covers.
    defender = BLACK if by_colour == WHITE else WHITE
    if PAWN_ATT[defender][square] & bb[P] & own:
        return True
    if KNIGHT_ATT[square] & bb[N] & own:
        return True
    if KING_ATT[square] & bb[K] & own:
        return True
    diagonal = bishop_attacks(square, occupied)
    if diagonal & (bb[B] | bb[Q]) & own:
        return True
    straight = rook_attacks(square, occupied)
    if straight & (bb[R] | bb[Q]) & own:
        return True
    return False


@njit(cache=False)
def piece_at(bb: np.ndarray, square: np.int64) -> np.int64:
    """Piece type on a square, or -1 if empty."""
    mask = ONE << U64(square)
    if not ((bb[WOCC] | bb[BOCC]) & mask):
        return -1
    for piece in range(6):
        if bb[piece] & mask:
            return piece
    return -1


@njit(cache=False)
def encode(frm: np.int64, to: np.int64, promo: np.int64, flag: np.int64) -> np.int64:
    return frm | (to << 6) | (promo << 12) | (flag << 15)


@njit(cache=False)
def generate_moves(bb: np.ndarray, st: np.ndarray, moves: np.ndarray) -> np.int64:
    """Pseudo-legal moves into `moves`. Returns how many were written.

    Legality is settled by make_move, which rejects a move that leaves the
    mover's own king attacked. Generating pseudo-legally and filtering there
    is both simpler and, at this node rate, faster than pin detection.
    """
    count = 0
    side = st[STM]
    own = bb[WOCC] if side == WHITE else bb[BOCC]
    enemy = bb[BOCC] if side == WHITE else bb[WOCC]
    occupied = own | enemy
    empty = ~occupied

    # ---- pawns ----
    pawns = bb[P] & own
    if side == WHITE:
        single = (pawns << U64(8)) & empty
        double = ((single & U64(0x0000000000FF0000)) << U64(8)) & empty
        promo_rank = U64(0xFF00000000000000)
        push_delta = -8
    else:
        single = (pawns >> U64(8)) & empty
        double = ((single & U64(0x0000FF0000000000)) >> U64(8)) & empty
        promo_rank = U64(0x00000000000000FF)
        push_delta = 8

    pushes = single
    while pushes:
        to = lsb(pushes)
        pushes &= pushes - ONE
        frm = to + push_delta
        if (ONE << U64(to)) & promo_rank:
            for promo in range(1, 5):
                moves[count] = encode(frm, to, promo, 0)
                count += 1
        else:
            moves[count] = encode(frm, to, 0, 0)
            count += 1

    doubles = double
    while doubles:
        to = lsb(doubles)
        doubles &= doubles - ONE
        moves[count] = encode(to + 2 * push_delta, to, 0, FLAG_DOUBLE)
        count += 1

    captures = pawns
    while captures:
        frm = lsb(captures)
        captures &= captures - ONE
        targets = PAWN_ATT[side][frm] & enemy
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            if (ONE << U64(to)) & promo_rank:
                for promo in range(1, 5):
                    moves[count] = encode(frm, to, promo, 0)
                    count += 1
            else:
                moves[count] = encode(frm, to, 0, 0)
                count += 1
        if st[EP] >= 0 and (PAWN_ATT[side][frm] & (ONE << U64(st[EP]))):
            moves[count] = encode(frm, st[EP], 0, FLAG_EP)
            count += 1

    # ---- knights ----
    knights = bb[N] & own
    while knights:
        frm = lsb(knights)
        knights &= knights - ONE
        targets = KNIGHT_ATT[frm] & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[count] = encode(frm, to, 0, 0)
            count += 1

    # ---- bishops and queens (diagonal) ----
    diagonals = (bb[B] | bb[Q]) & own
    while diagonals:
        frm = lsb(diagonals)
        diagonals &= diagonals - ONE
        targets = bishop_attacks(frm, occupied) & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[count] = encode(frm, to, 0, 0)
            count += 1

    # ---- rooks and queens (straight) ----
    straights = (bb[R] | bb[Q]) & own
    while straights:
        frm = lsb(straights)
        straights &= straights - ONE
        targets = rook_attacks(frm, occupied) & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[count] = encode(frm, to, 0, 0)
            count += 1

    # ---- king ----
    kings = bb[K] & own
    if kings:
        frm = lsb(kings)
        targets = KING_ATT[frm] & ~own
        while targets:
            to = lsb(targets)
            targets &= targets - ONE
            moves[count] = encode(frm, to, 0, 0)
            count += 1

        # ---- castling ----
        # Squares between must be empty, and the king must not start in, pass
        # through, or land on an attacked square.
        enemy_colour = BLACK if side == WHITE else WHITE
        if side == WHITE:
            if (
                (st[CASTLE] & WK)
                and not (occupied & U64(0x0000000000000060))
                and not attacked_by(bb, 4, enemy_colour)
                and not attacked_by(bb, 5, enemy_colour)
                and not attacked_by(bb, 6, enemy_colour)
            ):
                moves[count] = encode(4, 6, 0, FLAG_CASTLE)
                count += 1
            if (
                (st[CASTLE] & WQ)
                and not (occupied & U64(0x000000000000000E))
                and not attacked_by(bb, 4, enemy_colour)
                and not attacked_by(bb, 3, enemy_colour)
                and not attacked_by(bb, 2, enemy_colour)
            ):
                moves[count] = encode(4, 2, 0, FLAG_CASTLE)
                count += 1
        else:
            if (
                (st[CASTLE] & BK)
                and not (occupied & U64(0x6000000000000000))
                and not attacked_by(bb, 60, enemy_colour)
                and not attacked_by(bb, 61, enemy_colour)
                and not attacked_by(bb, 62, enemy_colour)
            ):
                moves[count] = encode(60, 62, 0, FLAG_CASTLE)
                count += 1
            if (
                (st[CASTLE] & BQ)
                and not (occupied & U64(0x0E00000000000000))
                and not attacked_by(bb, 60, enemy_colour)
                and not attacked_by(bb, 59, enemy_colour)
                and not attacked_by(bb, 58, enemy_colour)
            ):
                moves[count] = encode(60, 58, 0, FLAG_CASTLE)
                count += 1

    return count


# Castling rights are cleared whenever a king or rook leaves, or a rook is
# captured on, one of these squares.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 & ~WQ
CASTLE_MASK[4] = 15 & ~(WK | WQ)
CASTLE_MASK[7] = 15 & ~WK
CASTLE_MASK[56] = 15 & ~BQ
CASTLE_MASK[60] = 15 & ~(BK | BQ)
CASTLE_MASK[63] = 15 & ~BK


@njit(cache=False)
def _clear(bb: np.ndarray, piece: np.int64, colour: np.int64, square: np.int64) -> None:
    mask = ~(ONE << U64(square))
    bb[piece] &= mask
    bb[WOCC if colour == WHITE else BOCC] &= mask


@njit(cache=False)
def _set(bb: np.ndarray, piece: np.int64, colour: np.int64, square: np.int64) -> None:
    mask = ONE << U64(square)
    bb[piece] |= mask
    bb[WOCC if colour == WHITE else BOCC] |= mask


@njit(cache=False)
def make_move(
    bb: np.ndarray, st: np.ndarray, move: np.int64, undo: np.ndarray, ply: np.int64
) -> bool:
    """Play `move`. Returns False and takes it back if it was illegal.

    Illegal here means only one thing: it left our own king attacked.
    """
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7

    side = st[STM]
    enemy_colour = BLACK if side == WHITE else WHITE
    piece = piece_at(bb, frm)

    captured = -1
    capture_square = to
    if flag == FLAG_EP:
        capture_square = to - 8 if side == WHITE else to + 8
        captured = P
    else:
        captured = piece_at(bb, to)

    # Save everything unmake needs.
    undo[ply, 0] = captured
    undo[ply, 1] = capture_square
    undo[ply, 2] = st[CASTLE]
    undo[ply, 3] = st[EP]
    undo[ply, 4] = st[HALF]
    undo[ply, 5] = piece

    if captured >= 0:
        _clear(bb, captured, enemy_colour, capture_square)

    _clear(bb, piece, side, frm)
    if promo:
        _set(bb, promo, side, to)
    else:
        _set(bb, piece, side, to)

    if flag == FLAG_CASTLE:
        if to == 6:
            _clear(bb, R, WHITE, 7)
            _set(bb, R, WHITE, 5)
        elif to == 2:
            _clear(bb, R, WHITE, 0)
            _set(bb, R, WHITE, 3)
        elif to == 62:
            _clear(bb, R, BLACK, 63)
            _set(bb, R, BLACK, 61)
        else:
            _clear(bb, R, BLACK, 56)
            _set(bb, R, BLACK, 59)

    st[CASTLE] = st[CASTLE] & CASTLE_MASK[frm] & CASTLE_MASK[to]
    st[EP] = (frm + to) // 2 if flag == FLAG_DOUBLE else -1
    st[HALF] = 0 if (piece == P or captured >= 0) else st[HALF] + 1
    st[STM] = enemy_colour

    # Legality: our king must not be attacked now.
    king_bb = bb[K] & (bb[WOCC] if side == WHITE else bb[BOCC])
    if king_bb and attacked_by(bb, lsb(king_bb), enemy_colour):
        unmake_move(bb, st, move, undo, ply)
        return False
    return True


@njit(cache=False)
def unmake_move(
    bb: np.ndarray, st: np.ndarray, move: np.int64, undo: np.ndarray, ply: np.int64
) -> None:
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    flag = (move >> 15) & 7

    captured = undo[ply, 0]
    capture_square = undo[ply, 1]
    piece = undo[ply, 5]

    side = BLACK if st[STM] == WHITE else WHITE
    enemy_colour = st[STM]

    if promo:
        _clear(bb, promo, side, to)
    else:
        _clear(bb, piece, side, to)
    _set(bb, piece, side, frm)

    if captured >= 0:
        _set(bb, captured, enemy_colour, capture_square)

    if flag == FLAG_CASTLE:
        if to == 6:
            _clear(bb, R, WHITE, 5)
            _set(bb, R, WHITE, 7)
        elif to == 2:
            _clear(bb, R, WHITE, 3)
            _set(bb, R, WHITE, 0)
        elif to == 62:
            _clear(bb, R, BLACK, 61)
            _set(bb, R, BLACK, 63)
        else:
            _clear(bb, R, BLACK, 59)
            _set(bb, R, BLACK, 56)

    st[CASTLE] = undo[ply, 2]
    st[EP] = undo[ply, 3]
    st[HALF] = undo[ply, 4]
    st[STM] = side


@njit(cache=False)
def perft(
    bb: np.ndarray, st: np.ndarray, depth: np.int64, moves: np.ndarray, undo: np.ndarray, ply: np.int64
) -> np.int64:
    if depth == 0:
        return 1
    count = generate_moves(bb, st, moves[ply])
    total = 0
    for index in range(count):
        move = moves[ply, index]
        if make_move(bb, st, move, undo, ply):
            total += perft(bb, st, depth - 1, moves, undo, ply + 1)
            unmake_move(bb, st, move, undo, ply)
    return total
