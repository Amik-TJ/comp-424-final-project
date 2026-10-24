"""Constants and rule flags of the Goblet Gobblers game."""
from __future__ import annotations


# ---------------------------------------------------------------- the board

BOARD_SIZE = 4              # cells per side: 4 x 4 = 16 cells
NUM_SIZES = 3               # piece sizes: small, medium, large

# Piece sizes. A larger piece has a larger number, so "can this piece cover that one?" is a
# comparison: SMALL < MEDIUM < LARGE. The sizes also index the arrays:
#   board[r, c, LARGE]  -> the large slot of cell (r, c)
#   reserves[1, SMALL]  -> the number of small pieces player 1 has left
# Write the names, not the digits, so that a size is never confused with a player number.
SMALL, MEDIUM, LARGE = 0, 1, 2
SIZE_NAMES = ("S", "M", "L")    # letters used when a piece is printed, e.g. "2S" = player 2's small

# The values in the board array:
#   EMPTY    - no piece in this slot
#   PLAYER_1 - a piece of player 1
#   PLAYER_2 - a piece of player 2
#   OBSTACLE - the obstacle; all three slots of the obstacle cell hold this value
EMPTY, PLAYER_1, PLAYER_2, OBSTACLE = 0, 1, 2, 3
PLAYER_NAMES = {1: "Blue", 2: "Brown"}  # names shown on screen and in logs


# ---------------------------------------------------------------- the pieces of each player

INITIAL_RESERVE = (3, 3, 3)  # pieces per player before the removal: 3 small, 3 medium, 3 large
PIECES_REMOVED = 1           # pieces of one random size removed from both players, leaving 8 each


# ---------------------------------------------------------------- rule flags
# Set only by the framework, through configure_rules(). Agents read them and never assign them.

ALLOW_MOVES = True           # True: a turn may also move your own top piece to another cell.
                             # False: a turn can only place a piece from your reserve.
WIN_LENGTH = 4               # pieces in a row (row, column or diagonal) needed to win: 3 or 4


# ---------------------------------------------------------------- matches

ROUNDS_PER_MATCH = 2         # games in one match; always even, so each agent moves first in half of them


# ---------------------------------------------------------------- game length

MAX_PLIES = 100              # a ply is one player's turn, so 100 plies = 50 turns each.
                             # A game that nobody has won after 100 plies is a draw.


# ---------------------------------------------------------------- limits for each agent

STEP_TIME_LIMIT = 2.0        # seconds for one step(). A slower step() is a strike: its move is
                             # discarded and a random legal move is played instead.
STEP_HARD_LIMIT = 10.0       # seconds. A step() still running after this (e.g. an endless loop) is
                             # killed, and its agent loses the game.
INIT_TIME_LIMIT = 30.0       # seconds for the agent's __init__, once per game. A slower agent loses
                             # the game before the first move.
MAX_STRIKES = 3              # strikes forgiven per game; the 4th loses the game. A strike is an
                             # illegal move, an exception in step(), or a step() slower than STEP_TIME_LIMIT.
MEMORY_LIMIT_MB = 500        # megabytes of memory per agent process. The sandbox allows 200 MB more
                             # for Python and numpy.
LOG_CAP_BYTES = 1_000_000    # bytes (about 1 MB) of printed output kept per agent per game; the rest
                             # is discarded.


# ---------------------------------------------------------------- names of the rule flags

RULE_FLAGS = ("allow_moves", "win_length")   # the keys of the rules dict, e.g. {"allow_moves": True, "win_length": 3}


def configure_rules(*, allow_moves: bool | None = None, win_length: int | None = None) -> None:
    """Set the rule flags of this process; None leaves a flag unchanged.

    Takes:  allow_moves - True or False.
            win_length  - 3 or 4.
    Raises ValueError for any other win_length.
    The framework calls it once, before any agent module is imported. Agents must never call it.
    """
    global ALLOW_MOVES, WIN_LENGTH
    if allow_moves is not None:
        ALLOW_MOVES = bool(allow_moves)
    if win_length is not None:
        if win_length not in (3, 4):
            raise ValueError(f"win_length must be 3 or 4, not {win_length}.")
        WIN_LENGTH = win_length


def current_rules() -> dict:
    """Return the current rule flags as a dict, e.g. {"allow_moves": True, "win_length": 3}."""
    return {"allow_moves": ALLOW_MOVES, "win_length": WIN_LENGTH}
