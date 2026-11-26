"""Move-ordering scores and killer/history heuristics."""

NUM_CELLS = 16
NUM_SIZES = 3
PLAYER_2 = 2
PLACEMENT_SRC = -1
EMPTY = 0
MAX_KILLER_PLY = 48
MAX_HISTORY_SCORE = 1_000_000

ORDER_TT_MOVE = 50_000_000
ORDER_WIN_THREAT = 20_000_000
ORDER_KILLER = 5_000_000
ORDER_BLOCK_THREAT = 3_000_000
ORDER_CAPTURE_BASE = 4000
ORDER_CAPTURE_TARGET_SIZE = 500
ORDER_CAPTURE_SELF_SIZE_PENALTY = 40
ORDER_EMPTY_BASE = 300
ORDER_EMPTY_SIZE_BONUS = 20
ORDER_SELF_STACK = 40
ORDER_DEST_HEAT_BONUS = 35
ORDER_SRC_HEAT_PENALTY = 20
ORDER_PLACEMENT_BONUS = 80


class MoveOrderingMixin:
    """Initializes and applies the search's move-ordering heuristics."""

    def _reset_move_ordering(self):
        self.killers = [[None, None] for _ in range(MAX_KILLER_PLY)]
        self.hist = [
            [[[0] * NUM_SIZES for _ in range(NUM_CELLS)] for _ in range(NUM_CELLS + 1)]
            for _ in range(PLAYER_2 + 1)
        ]

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
