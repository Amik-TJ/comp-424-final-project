"""Evaluation logic kept separate from the search algorithm."""

BOARD_SIZE = 4
NUM_SIZES = 3
PIECE_SIZES = (0, 1, 2)
LARGE_SIZE = 2
NO_SIZE = -1
MATE = 100_000
DOUBLE_THREAT_MIN = 2
TOP_PIECE_VALUES = (16, 34, 58)
BURIED_PIECE_VALUES = (3, 6, 10)
RESERVE_PIECE_VALUES = (6, 16, 40)
OPEN_LINE_SCORES = {1: 4800, 2: 170, 3: 26}
EVAL_HEAT_BASE = 4
EVAL_MULTI_THREAT_PENALTY = 22_000
EVAL_DOUBLE_THREAT_BONUS = 2800
EVAL_SCORE_CLAMP = 80_000
LINE_OPEN_SIZE_WEIGHT = 3
LINE_BLOCKED_THREAT_BASE = 900
LINE_BLOCKED_THREAT_SIZE_WEIGHT = 2
LINE_BLOCKED_SETUP_BASE = 60
LINE_CAPPED_WEIGHT = 5
LINE_CONTESTED_WEIGHT = 4

def open_points(n_self, win_len):
    return OPEN_LINE_SCORES.get(win_len - n_self, 0)

def evaluate(state, player):
    my_targets = state._threat_targets(player)
    if state._can_win(player, my_targets):
        return MATE - 1
    opp = 3 - player
    opp_targets = state._threat_targets(opp)
    score = material(state, player) - material(state, opp)
    score += lines(state, player) - lines(state, opp)
    if len(opp_targets) >= DOUBLE_THREAT_MIN:
        score -= EVAL_MULTI_THREAT_PENALTY
    return max(-EVAL_SCORE_CLAMP, min(EVAL_SCORE_CLAMP, score))

def material(state, player):
    score = 0
    for i in state.playable:
        top, owner, heat = state.sz[i], state.own[i], state.heat[i]
        if owner != player:
            continue
        base = i * NUM_SIZES
        for size in PIECE_SIZES:
            if state.cells[base + size] == player:
                if size == top:
                    score += TOP_PIECE_VALUES[size] + heat * (EVAL_HEAT_BASE + size)
                else:
                    score += BURIED_PIECE_VALUES[size]
    reserves = state.res[player]
    return score + sum(reserves[s] * RESERVE_PIECE_VALUES[s] for s in PIECE_SIZES)

def lines(state, player):
    opponent = 3 - player
    total = threats = 0
    for line_cells in state.line_cells:
        n_self = n_opp = size_sum = 0
        opponent_size = NO_SIZE
        for i in line_cells:
            owner = state.own[i]
            if owner == player:
                n_self += 1
                size_sum += state.sz[i] + 1
            elif owner == opponent:
                n_opp += 1
                opponent_size = state.sz[i]
        if n_self == 0:
            continue
        if n_opp == 0:
            total += open_points(n_self, state.W) + size_sum * LINE_OPEN_SIZE_WEIGHT
            if n_self == state.W - 1:
                threats += 1
        elif n_opp == 1 and opponent_size < LARGE_SIZE and n_self == state.W - 1:
            total += LINE_BLOCKED_THREAT_BASE + size_sum * LINE_BLOCKED_THREAT_SIZE_WEIGHT
            threats += 1
        elif n_opp == 1 and opponent_size < LARGE_SIZE and n_self == state.W - 2:
            total += LINE_BLOCKED_SETUP_BASE + size_sum
        elif n_opp == 1 and opponent_size >= LARGE_SIZE:
            total += LINE_CAPPED_WEIGHT * n_self
        elif n_self > n_opp:
            total += LINE_CONTESTED_WEIGHT * n_self
    if threats >= DOUBLE_THREAT_MIN:
        total += EVAL_DOUBLE_THREAT_BONUS
    return total
