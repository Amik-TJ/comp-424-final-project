"""Student agent: iterative-deepening negamax with alpha-beta pruning."""

import time

import constants
from agents.agent import Agent
from agents.game_state import COORDS, PLACEMENT_SRC, GameStateMixin
from agents.move_ordering import MoveOrderingMixin
from helpers import Move, check_move_validity, get_valid_moves
from store import register_agent
from agents import evaluation

USE_CUSTOM_MOVE_GENERATION = False

DEFAULT_MAX_PLIES = 100
TIME_MARGIN = 0.25
SOFT_TIME_RATIO = 0.45
MIN_SEARCH_BUDGET = 0.05
MAX_SEARCH_DEPTH = 32
TIME_CHECK_MASK = 63
ENDGAME_TT_PLY_THRESHOLD = 14
IMMEDIATE_WIN_DEPTH = 64
MATE = 100_000
MATE_TH = 90_000
INF = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2


class _TimeUp(Exception):
    """Raised when the search reaches its time deadline."""


@register_agent("student_agent")
class StudentAgent(GameStateMixin, MoveOrderingMixin, Agent):
    """Runs iterative deepening and negamax alpha-beta on a compact board state."""

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.use_custom_move_generation = USE_CUSTOM_MOVE_GENERATION
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
                else:
                    val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, 1, self.plies_left - 1))
            finally:
                self.undo(token)
            if won:
                self.tt[key] = (depth, MATE - 1, EXACT, m)
                return MATE - 1, m
            if revealed:
                continue
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
            return evaluation.evaluate(self, player)
        my_targets = self._threat_targets(player)
        if self._can_win(player, my_targets):
            self.tt[key] = (IMMEDIATE_WIN_DEPTH, MATE - 1, EXACT, tt_move)
            return MATE - 1
        moves = self._gen_moves(player)
        if not moves:
            return -(MATE - 1)
        moves = self._order(player, moves, tt_move, ply, my_targets)
        best, best_move = -INF, None
        opp = 3 - player
        for m in moves:
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
                else:
                    val = self._adjust(-self._negamax(depth - 1, -beta, -alpha, opp, ply + 1, plies_left - 1))
            finally:
                self.undo(token)
            if won:
                self._killer(ply, m)
                self.tt[key] = (depth, MATE - 1, EXACT, m)
                return MATE - 1
            if revealed:
                continue
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

    def _as_move(self, packed):
        src, dest, size = packed
        start = COORDS[src] if src != PLACEMENT_SRC else None
        return Move(start, COORDS[dest], size)
