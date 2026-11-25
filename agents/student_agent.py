"""Student agent"""

import time

import constants
from agents.agent import Agent
from helpers import LINES, Move, check_move_validity, get_valid_moves
from store import register_agent

BOARD_SIZE = 4
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_SIZES = 3
NUM_SLOTS = NUM_CELLS * NUM_SIZES
PIECE_SIZES = (0, 1, 2)
LARGE_SIZE = 2
NO_SIZE = -1
PLACEMENT_SRC = -1

EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2
OBSTACLE = 3
DEFAULT_MAX_PLIES = 100

BITS_PER_ENTRY = 2
ENTRY_MASK = (1 << BITS_PER_ENTRY) - 1
RES_BIT_OFFSET = NUM_SLOTS * BITS_PER_ENTRY
PLAYER_BIT_OFFSET = RES_BIT_OFFSET + 2 * NUM_SIZES * BITS_PER_ENTRY
PLAYER_BIT = 1 << PLAYER_BIT_OFFSET

TIME_MARGIN = 0.25
SOFT_TIME_RATIO = 0.45
MIN_SEARCH_BUDGET = 0.05
MAX_SEARCH_DEPTH = 32
MAX_TT_ENTRIES = 250_000
MAX_KILLER_PLY = 48
TIME_CHECK_MASK = 63
ENDGAME_TT_PLY_THRESHOLD = 14
IMMEDIATE_WIN_DEPTH = 64

LMR_MIN_DEPTH = 3
LMR_MIN_MOVES_TRIED = 4
LMR_REDUCTION = 1

MATE = 100_000
MATE_TH = 90_000
INF = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2

ORDER_TT_MOVE = 50_000_000
ORDER_WIN_THREAT = 20_000_000
ORDER_KILLER = 5_000_000
ORDER_BLOCK_THREAT = 3_000_000
MAX_HISTORY_SCORE = 1_000_000

ORDER_CAPTURE_BASE = 4000
ORDER_CAPTURE_TARGET_SIZE = 500
ORDER_CAPTURE_SELF_SIZE_PENALTY = 40
ORDER_EMPTY_BASE = 300
ORDER_EMPTY_SIZE_BONUS = 20
ORDER_SELF_STACK = 40
ORDER_DEST_HEAT_BONUS = 35
ORDER_SRC_HEAT_PENALTY = 20
ORDER_PLACEMENT_BONUS = 80

TOP_PIECE_VALUES = (16, 34, 58)
BURIED_PIECE_VALUES = (3, 6, 10)
RESERVE_PIECE_VALUES = (6, 16, 40)
OPEN_LINE_SCORES = {1: 4800, 2: 170, 3: 26}

EVAL_HEAT_BASE = 4
EVAL_MULTI_THREAT_PENALTY = 22_000
EVAL_DOUBLE_THREAT_BONUS = 2800
EVAL_SCORE_CLAMP = 80_000
DOUBLE_THREAT_MIN = 2

LINE_OPEN_SIZE_WEIGHT = 3
LINE_BLOCKED_THREAT_BASE = 900
LINE_BLOCKED_THREAT_SIZE_WEIGHT = 2
LINE_BLOCKED_SETUP_BASE = 60
LINE_CAPPED_WEIGHT = 5
LINE_CONTESTED_WEIGHT = 4

COORDS = tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))


class _TimeUp(Exception):
    pass


def _open_points(n_self, win_len):
    return OPEN_LINE_SCORES.get(win_len - n_self, 0)


@register_agent("student_agent")  # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.last_depth = 0
        self.last_score = 0
        self.last_nodes = 0
        self.turn_count = 0
        self.tt = {}
        self._tt_sig = None

    def step(self, board, reserves, player, opponent):
        start = time.perf_counter()
        saved_b = board.copy()
        saved_r = reserves.copy()

        current_ply = 2 * self.turn_count + (int(player) - 1)
        self.turn_count += 1
        max_plies = int(getattr(constants, "MAX_PLIES", DEFAULT_MAX_PLIES))

        move = self._choose(board, reserves, player, start, max(0, max_plies - current_ply))
        if move is not None and check_move_validity(saved_b, saved_r, player, move):
            return move

        legal = get_valid_moves(saved_b, saved_r, player)
        if not legal:
            raise RuntimeError("step() was called when the player has no legal moves")
        return legal[0]

    def _choose(self, board, reserves, player, start, plies_left=DEFAULT_MAX_PLIES):
        self._prepare(board, reserves, player)
        self.plies_left = max(0, int(plies_left))
        budget = max(MIN_SEARCH_BUDGET, constants.STEP_TIME_LIMIT - TIME_MARGIN)
        self.deadline = start + budget
        soft_deadline = start + budget * SOFT_TIME_RATIO

        moves = self._gen_moves(player)
        if not moves:
            raise RuntimeError("no legal moves")

        ordered = self._order(player, moves, None, 0)
        best = ordered[0]
        self.last_depth = 0

        for depth in range(1, min(MAX_SEARCH_DEPTH, self.plies_left + 1)):
            now = time.perf_counter()
            if now >= self.deadline or (depth > 1 and now >= soft_deadline):
                break

            self._root_partial = None
            try:
                score, move = self._search_root(depth, player, ordered)
            except _TimeUp:
                if self._root_partial is not None:
                    score, move = self._root_partial
                    if move is not None:
                        best = move
                        self.last_score = score
                break

            if move is not None:
                best = move
                self.last_depth = depth
                self.last_score = score
            if score is not None and abs(score) > MATE_TH:
                break
            ordered = self._order(player, moves, best, 0)

        self.last_nodes = self.nodes
        return self._as_move(best)

    def _prepare(self, board, reserves, player):
        self.allow_moves = bool(constants.ALLOW_MOVES)
        self.W = int(constants.WIN_LENGTH)

        self.cells = [EMPTY] * NUM_SLOTS
        self.res = [
            [0] * NUM_SIZES,
            [int(reserves[PLAYER_1, s]) for s in PIECE_SIZES],
            [int(reserves[PLAYER_2, s]) for s in PIECE_SIZES],
        ]
        self.own = [EMPTY] * NUM_CELLS
        self.sz = [NO_SIZE] * NUM_CELLS
        self.bb = [0] * (OBSTACLE + 1)
        self.playable = []
        self.z = 0
        obstacle = NO_SIZE

        for i, (r, c) in enumerate(COORDS):
            base = i * NUM_SIZES
            s0 = int(board[r, c, 0])
            s1 = int(board[r, c, 1])
            s2 = int(board[r, c, 2])
            self.cells[base] = s0
            self.cells[base + 1] = s1
            self.cells[base + 2] = s2

            if s0 == OBSTACLE:
                self.own[i] = OBSTACLE
                self.bb[OBSTACLE] |= 1 << i
                obstacle = i
            else:
                self.playable.append(i)
                if s2:
                    top_owner, top_size = s2, 2
                elif s1:
                    top_owner, top_size = s1, 1
                elif s0:
                    top_owner, top_size = s0, 0
                else:
                    top_owner, top_size = EMPTY, NO_SIZE
                self.own[i] = top_owner
                self.sz[i] = top_size
                self.bb[top_owner] |= 1 << i

            self.z |= (
                (s0 << (base * BITS_PER_ENTRY))
                | (s1 << ((base + 1) * BITS_PER_ENTRY))
                | (s2 << ((base + 2) * BITS_PER_ENTRY))
            )

        for p in (PLAYER_1, PLAYER_2):
            for s in PIECE_SIZES:
                self._hash_write_res(p, s)
        if player == PLAYER_2:
            self.z |= PLAYER_BIT

        sig = (self.W, self.allow_moves, obstacle)
        if sig != self._tt_sig or len(self.tt) > MAX_TT_ENTRIES:
            self.tt.clear()
            self._tt_sig = sig

        self.masks = []
        self.line_cells = []
        self.heat = [0] * NUM_CELLS

        for line in LINES[self.W]:
            idxs = [r * BOARD_SIZE + c for r, c in line]
            if obstacle in idxs:
                continue
            mask = 0
            for i in idxs:
                mask |= 1 << i
                self.heat[i] += 1
            self.masks.append(mask)
            self.line_cells.append(tuple(idxs))

        self.killers = [[None, None] for _ in range(MAX_KILLER_PLY)]
        self.hist = [
            [[[0] * NUM_SIZES for _ in range(NUM_CELLS)] for _ in range(NUM_CELLS + 1)]
            for _ in range(PLAYER_2 + 1)
        ]
        self.nodes = 0

    def _hash_write_slot(self, idx, val):
        shift = idx * BITS_PER_ENTRY
        self.z = (self.z & ~(ENTRY_MASK << shift)) | (val << shift)

    def _hash_write_res(self, player, size):
        shift = RES_BIT_OFFSET + ((player - 1) * NUM_SIZES + size) * BITS_PER_ENTRY
        self.z = (self.z & ~(ENTRY_MASK << shift)) | (self.res[player][size] << shift)

    def _set_top(self, i, owner, size):
        old = self.own[i]
        if old != owner:
            bit = 1 << i
            self.bb[old] &= ~bit
            self.bb[owner] |= bit
            self.own[i] = owner
        self.sz[i] = size

    def apply(self, player, src, dest, size):
        """Apply move in place (src == PLACEMENT_SRC for placement) and return undo token."""
        old_od, old_sd = self.own[dest], self.sz[dest]
        old_os, old_ss = EMPTY, NO_SIZE

        if src == PLACEMENT_SRC:
            self.res[player][size] -= 1
            self._hash_write_res(player, size)
        else:
            src_base = src * NUM_SIZES
            old_os, old_ss = self.own[src], self.sz[src]
            self.cells[src_base + size] = EMPTY
            self._hash_write_slot(src_base + size, EMPTY)

            no, ns = EMPTY, NO_SIZE
            for s in range(size - 1, -1, -1):
                v = self.cells[src_base + s]
                if v:
                    no, ns = v, s
                    break
            self._set_top(src, no, ns)

        dest_idx = dest * NUM_SIZES + size
        self.cells[dest_idx] = player
        self._hash_write_slot(dest_idx, player)
        self._set_top(dest, player, size)
        self.z ^= PLAYER_BIT
        return (src, dest, size, player, old_od, old_sd, old_os, old_ss)

    def undo(self, token):
        src, dest, size, player, old_od, old_sd, old_os, old_ss = token
        dest_idx = dest * NUM_SIZES + size
        self.cells[dest_idx] = EMPTY
        self._hash_write_slot(dest_idx, EMPTY)
        self._set_top(dest, old_od, old_sd)

        if src == PLACEMENT_SRC:
            self.res[player][size] += 1
            self._hash_write_res(player, size)
        else:
            src_idx = src * NUM_SIZES + size
            self.cells[src_idx] = player
            self._hash_write_slot(src_idx, player)
            self._set_top(src, old_os, old_ss)

        self.z ^= PLAYER_BIT

    def _gen_moves(self, player):
        sz, own, pres, playable = self.sz, self.own, self.res[player], self.playable
        moves = [
            (PLACEMENT_SRC, d, s)
            for s in PIECE_SIZES
            if pres[s] > 0
            for d in playable
            if sz[d] < s
        ]
        if self.allow_moves:
            for src in playable:
                if own[src] == player:
                    s = sz[src]
                    moves.extend((src, d, s) for d in playable if d != src and sz[d] < s)
        return moves

    def _terminal(self, mover):
        """Return winner after mover's move (checking opponent first for uncovered lines), or EMPTY."""
        for p in (3 - mover, mover):
            bb = self.bb[p]
            if any((bb & m) == m for m in self.masks):
                return p
        return EMPTY

    def _threat_targets(self, player):
        """Cells that would complete a line for player if occupied."""
        out = []
        seen = 0
        need = self.W - 1
        bb = self.bb[player]
        opp = 3 - player

        for m in self.masks:
            if (bb & m).bit_count() != need:
                continue
            i = (m & ~bb).bit_length() - 1
            bit = 1 << i
            if seen & bit or (self.own[i] == opp and self.sz[i] >= LARGE_SIZE):
                continue
            seen |= bit
            out.append(i)
        return out

    def _can_win(self, player, targets=None):
        if targets is None:
            targets = self._threat_targets(player)
        if not targets:
            return False

        sz, pres = self.sz, self.res[player]
        for t in targets:
            ts = sz[t]
            if any(pres[s] > 0 and ts < s for s in PIECE_SIZES):
                return True

        if not self.allow_moves:
            return False

        for src in self.playable:
            if self.own[src] == player:
                s = sz[src]
                for t in targets:
                    if t != src and sz[t] < s and self._move_wins(player, src, t, s):
                        return True
        return False

    def _move_wins(self, player, src, dest, size):
        token = self.apply(player, src, dest, size)
        try:
            return self._terminal(player) == player
        finally:
            self.undo(token)

    def _order(self, player, moves, tt_move, ply, my_targets=None):
        opp = 3 - player
        my_t = set(self._threat_targets(player) if my_targets is None else my_targets)
        opp_t = set(self._threat_targets(opp))
        heat, own, sz = self.heat, self.own, self.sz
        hist = self.hist[player]
        killer = self.killers[ply] if ply < len(self.killers) else (None, None)

        scored = []
        for m in moves:
            src, dest, size = m
            sc = 0
            if tt_move is not None and m == tt_move:
                sc += ORDER_TT_MOVE
            if dest in my_t:
                sc += ORDER_WIN_THREAT
            if m == killer[0] or m == killer[1]:
                sc += ORDER_KILLER
            if dest in opp_t:
                sc += ORDER_BLOCK_THREAT

            od = own[dest]
            if od == opp:
                sc += (
                    ORDER_CAPTURE_BASE
                    + sz[dest] * ORDER_CAPTURE_TARGET_SIZE
                    - size * ORDER_CAPTURE_SELF_SIZE_PENALTY
                )
            elif od == EMPTY:
                sc += ORDER_EMPTY_BASE + size * ORDER_EMPTY_SIZE_BONUS
            else:
                sc += ORDER_SELF_STACK

            sc += heat[dest] * ORDER_DEST_HEAT_BONUS + (
                -heat[src] * ORDER_SRC_HEAT_PENALTY
                if src != PLACEMENT_SRC
                else ORDER_PLACEMENT_BONUS
            )
            sc += min(hist[src + 1][dest][size], MAX_HISTORY_SCORE)
            scored.append((sc, m))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored]

    def _killer(self, ply, move):
        if ply < len(self.killers) and self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def _bump_hist(self, player, move, depth):
        src, dest, size = move
        row = self.hist[player][src + 1][dest]
        row[size] = min(row[size] + depth * depth, MAX_HISTORY_SCORE)

    def _adjust(self, val):
        if val > MATE_TH:
            return val - 1
        if val < -MATE_TH:
            return val + 1
        return val

    def _tt_key(self, plies_left):
        return self.z if plies_left > ENDGAME_TT_PLY_THRESHOLD else (self.z, plies_left)

    def _search_root(self, depth, player, moves):
        alpha, beta = -INF, INF
        alpha0 = alpha
        best, best_move = -INF, None
        opp = 3 - player
        tried = 0
        key = self._tt_key(self.plies_left)

        for m in moves:
            if time.perf_counter() >= self.deadline:
                raise _TimeUp

            token = self.apply(player, *m)
            won = revealed = False
            val = 0
            try:
                w = self._terminal(player)
                if w == player:
                    won = True
                elif w == opp:
                    revealed = True
                elif self.plies_left <= 1:
                    val = 0
                elif tried == 0:
                    val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, 1, self.plies_left - 1))
                else:
                    val = self._adjust(-self._negamax(depth - 1, -alpha - 1, -alpha, opp, 1, self.plies_left - 1))
                    if alpha < val < beta and time.perf_counter() < self.deadline:
                        val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, 1, self.plies_left - 1))
            finally:
                self.undo(token)

            if won:
                self.tt[key] = (depth, MATE - 1, EXACT, m)
                return MATE - 1, m
            if revealed:
                continue

            tried += 1
            if val > best:
                best, best_move = val, m
            if val > alpha:
                alpha = val
            self._root_partial = (best, best_move)

        if best_move is None:
            self.tt[key] = (depth, -(MATE - 1), EXACT, moves[0])
            return -(MATE - 1), moves[0]

        flag = UPPER if best <= alpha0 else (LOWER if best >= beta else EXACT)
        self.tt[key] = (depth, best, flag, best_move)
        return best, best_move

    def _negamax(self, depth, alpha, beta, player, ply, plies_left):
        self.nodes += 1
        if (self.nodes & TIME_CHECK_MASK) == 0 and time.perf_counter() >= self.deadline:
            raise _TimeUp
        if plies_left <= 0:
            return 0

        alpha0 = alpha
        key = self._tt_key(plies_left)
        tt = self.tt.get(key)
        tt_move = None
        if tt is not None:
            tdepth, tscore, tflag, tt_move = tt
            if tdepth >= depth:
                if tflag == EXACT or (tflag == LOWER and tscore >= beta) or (tflag == UPPER and tscore <= alpha):
                    return tscore

        if depth <= 0:
            return self._evaluate(player)

        my_targets = self._threat_targets(player)
        if self._can_win(player, my_targets):
            self.tt[key] = (IMMEDIATE_WIN_DEPTH, MATE - 1, EXACT, tt_move)
            return MATE - 1

        moves = self._gen_moves(player)
        if not moves:
            return -(MATE - 1)
        moves = self._order(player, moves, tt_move, ply, my_targets)

        best, best_move = -INF, None
        tried = 0
        opp = 3 - player
        killer = self.killers[ply] if ply < len(self.killers) else (None, None)

        for m in moves:
            dest = m[1]
            is_quiet = (
                self.own[dest] == EMPTY
                and m != tt_move
                and m != killer[0]
                and m != killer[1]
            )
            token = self.apply(player, *m)
            won = revealed = False
            val = 0
            try:
                w = self._terminal(player)
                if w == player:
                    won = True
                elif w == opp:
                    revealed = True
                elif plies_left <= 1:
                    val = 0
                elif tried == 0:
                    val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, ply + 1, plies_left - 1))
                else:
                    reduce = (
                        LMR_REDUCTION
                        if (
                            depth >= LMR_MIN_DEPTH
                            and tried >= LMR_MIN_MOVES_TRIED
                            and is_quiet
                            and abs(alpha) < MATE_TH
                        )
                        else 0
                    )
                    val = self._adjust(-self._negamax(depth - 1 - reduce, -alpha - 1, -alpha, opp, ply + 1, plies_left - 1))
                    if reduce and val > alpha:
                        val = self._adjust(-self._negamax(depth - 1, -alpha - 1, -alpha, opp, ply + 1, plies_left - 1))
                    if alpha < val < beta:
                        val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, ply + 1, plies_left - 1))
            finally:
                self.undo(token)

            if won:
                self._killer(ply, m)
                self.tt[key] = (depth, MATE - 1, EXACT, m)
                return MATE - 1
            if revealed:
                continue

            tried += 1
            if val > best:
                best, best_move = val, m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                self._killer(ply, m)
                self._bump_hist(player, m, depth)
                break

        if best_move is None:
            score = -(MATE - 1)
            self.tt[key] = (depth, score, EXACT, moves[0])
            return score

        flag = UPPER if best <= alpha0 else (LOWER if best >= beta else EXACT)
        self.tt[key] = (depth, best, flag, best_move)
        return best

    def _evaluate(self, player):
        my_targets = self._threat_targets(player)
        if self._can_win(player, my_targets):
            return MATE - 1

        opp = 3 - player
        opp_targets = self._threat_targets(opp)
        score = (self._material(player) - self._material(opp)) + (self._lines(player) - self._lines(opp))
        if len(opp_targets) >= DOUBLE_THREAT_MIN:
            score -= EVAL_MULTI_THREAT_PENALTY
        return max(-EVAL_SCORE_CLAMP, min(EVAL_SCORE_CLAMP, score))

    def _material(self, player):
        sc = 0
        cells, own, sz, heat = self.cells, self.own, self.sz, self.heat

        for i in self.playable:
            top, owner, h = sz[i], own[i], heat[i]
            if owner != player:
                continue
            base = i * NUM_SIZES
            for s in PIECE_SIZES:
                if cells[base + s] == player:
                    sc += (
                        (TOP_PIECE_VALUES[s] + h * (EVAL_HEAT_BASE + s))
                        if s == top
                        else BURIED_PIECE_VALUES[s]
                    )

        pres = self.res[player]
        return sc + sum(pres[s] * RESERVE_PIECE_VALUES[s] for s in PIECE_SIZES)

    def _lines(self, player):
        opp = 3 - player
        W, own, sz = self.W, self.own, self.sz
        total = threats = 0

        for cells in self.line_cells:
            n_self = n_opp = size_sum = 0
            opp_sz = NO_SIZE
            for i in cells:
                o = own[i]
                if o == player:
                    n_self += 1
                    size_sum += sz[i] + 1
                elif o == opp:
                    n_opp += 1
                    opp_sz = sz[i]

            if n_self == 0:
                continue

            if n_opp == 0:
                total += _open_points(n_self, W) + size_sum * LINE_OPEN_SIZE_WEIGHT
                if n_self == W - 1:
                    threats += 1
            elif n_opp == 1 and opp_sz < LARGE_SIZE and n_self == W - 1:
                total += LINE_BLOCKED_THREAT_BASE + size_sum * LINE_BLOCKED_THREAT_SIZE_WEIGHT
                threats += 1
            elif n_opp == 1 and opp_sz < LARGE_SIZE and n_self == W - 2:
                total += LINE_BLOCKED_SETUP_BASE + size_sum
            elif n_opp == 1 and opp_sz >= LARGE_SIZE:
                total += LINE_CAPPED_WEIGHT * n_self
            elif n_self > n_opp:
                total += LINE_CONTESTED_WEIGHT * n_self

        if threats >= DOUBLE_THREAT_MIN:
            total += EVAL_DOUBLE_THREAT_BONUS
        return total

    def _as_move(self, packed):
        src, dest, size = packed
        start = COORDS[src] if src != PLACEMENT_SRC else None
        return Move(start, COORDS[dest], size)
