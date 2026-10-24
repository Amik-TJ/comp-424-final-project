"""Constants and rule flags for the Goblet Gobblers game."""
from __future__ import annotations


# ---------------------------------------------------------------- the board

BOARD_SIZE = 4              # cells per side: the board is 4 x 4 = 16 cells
NUM_SIZES = 3               # how many piece sizes exist: small, medium, large

# Piece sizes. Each size is a number, so a bigger piece has a bigger number
# and "can this piece cover that one?" is a plain comparison: SMALL < MEDIUM < LARGE.
# The numbers are also positions in the arrays:
#   board[r, c, LARGE]  -> the large slot of cell (r, c)
#   reserves[1, SMALL]  -> how many smalls player 1 has left
# Always write the name, never the digit, so it is not confused with player 1 or 2.
SMALL, MEDIUM, LARGE = 0, 1, 2
SIZE_NAMES = ("S", "M", "L")    # letters used when a piece is printed, e.g. "2S" = player 2's small

# What a number in the board array means.
#   EMPTY    = nothing in this slot
#   PLAYER_1 = a piece of player 1 (player 1 always moves first)
#   PLAYER_2 = a piece of player 2
#   OBSTACLE = the blocked cell; all three slots of that one cell hold this value
EMPTY, PLAYER_1, PLAYER_2, OBSTACLE = 0, 1, 2, 3
PLAYER_NAMES = {1: "Blue", 2: "Brown"}  # names shown on screen and in logs


# ---------------------------------------------------------------- the pieces each player gets

INITIAL_RESERVE = (3, 3, 3)  # pieces per player before the draw: 3 small, 3 medium, 3 large
PIECES_REMOVED = 1           # pieces taken away at random at the start, the same size for both
                             # players, so each player plays with 3 + 3 + 3 - 1 = 8 pieces


# ---------------------------------------------------------------- rule flags 
# Only the framework changes these, through configure_rules(). Agents only read them.

ALLOW_MOVES = True           # True: a turn may also lift your own top piece and put it on another cell.
                             # False: a turn can only place a new piece from your reserve.
WIN_LENGTH = 4               # how many of your pieces in a row (row, column or diagonal) win: 3 or 4


# ---------------------------------------------------------------- game length

MAX_PLIES = 100              # unit: plies. A ply is one player's turn, so 100 plies = 50 turns each.
                             # If nobody has won after 100 plies, the game is a draw.


# ---------------------------------------------------------------- limits for each agent 

STEP_TIME_LIMIT = 2.0        # unit: seconds. The time step() should take to choose a move.
                             # If step() returns after 2 s but before STEP_HARD_LIMIT: one strike,
                             # the late move is thrown away and a random legal move is played instead.
STEP_HARD_LIMIT = 10.0       # unit: seconds. If step() has still not returned after 10 s (for example
                             # an endless loop), the agent's process is killed and that agent loses the game.
INIT_TIME_LIMIT = 30.0       # unit: seconds. Time the agent's __init__ may take, once per game.
                             # Slower: the agent loses the game before the first move.
MAX_STRIKES = 3              # unit: number of strikes. A strike is an illegal move, an error inside
                             # step(), or a step() slower than STEP_TIME_LIMIT. 3 strikes are
                             # forgiven in one game; the 4th strike loses the game.
MEMORY_LIMIT_MB = 500        # unit: megabytes (MB) of memory per agent process.
                             # The sandbox adds 200 MB on top for Python and numpy themselves.
LOG_CAP_BYTES = 1_000_000    # unit: bytes (1_000_000 bytes = about 1 MB). The most printed text
                             # kept per agent per game; anything printed after that is thrown away.


# ---------------------------------------------------------------- names of the rule flags

RULE_FLAGS = ("allow_moves", "win_length")   # the keys of the rules dict that is passed around,
                                             # e.g. {"allow_moves": True, "win_length": 3}


def configure_rules(*, allow_moves: bool | None = None, win_length: int | None = None) -> None:
    """
    Set the rule flags for this process (None =  leave unchanged). win_length must be 3 or 4.
    Called exactly once, by the game, before any agent module is imported. Agents must never call it. 
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