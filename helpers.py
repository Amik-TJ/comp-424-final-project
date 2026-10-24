"""The rules of Goblet Gobblers, COMP 424 variant.

The functions here only work things out and give back an answer: nothing is printed or saved,
and only execute_move changes the board you give it. The two rule flags, constants.ALLOW_MOVES
and constants.WIN_LENGTH, are read only when you leave allow_moves / win_length as None.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import constants
from constants import (BOARD_SIZE, NUM_SIZES, SMALL, MEDIUM, LARGE, EMPTY, PLAYER_1,
                       PLAYER_2, OBSTACLE, INITIAL_RESERVE, MAX_PLIES, SIZE_NAMES)




def _resolve_allow_moves(allow_moves: bool | None) -> bool:
    """Are board moves allowed for this call? Gives True or False.

    Takes: allow_moves - True, False, or None.
    None means "use the rule of this game": constants.ALLOW_MOVES. It is read again on
    every call, because the framework can change the rule after this file is loaded.
    """
    if allow_moves is None:
        return bool(constants.ALLOW_MOVES)        # the module's value, so always the current rule
    return bool(allow_moves)                      # bool() turns e.g. 1 into True



def _resolve_win_length(win_length: int | None) -> int:
    """How many pieces in a row win for this call? Gives 3 or 4.

    Takes: win_length - 3, 4, or None.
    None means "use the rule of this game": constants.WIN_LENGTH, read again on every call.
    Any other number raises ValueError.
    """
    if win_length is None:
        win_length = constants.WIN_LENGTH         # the module's value, so always the current rule
    if win_length not in (3, 4):
        raise ValueError(f"win_length must be 3 or 4, got {win_length!r}")
    return int(win_length)



def _build_lines(length: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Make the list of every way to win: all straight lines of `length` cells on the board.

    Takes: length - how many in a row win: 3 or 4.
    Gives: a tuple of lines. Each line is a tuple of (row, col) cells.
           Example line: ((0, 0), (0, 1), (0, 2)) = the first three cells of row 0.
           Rows, columns and both diagonals, each line once: 24 lines for 3, 10 lines for 4.
    Lines that cross the obstacle are kept. They can never win, because nobody owns the
    obstacle cell, so they need no special case.
    Runs only once, when this file is loaded, to fill LINES.
    """
    lines = []
    for r in range(BOARD_SIZE):                         # start from every cell ...
        for c in range(BOARD_SIZE):
            # ... and walk in 4 directions: right, down, down-right, down-left.
            # (Walking left, up, ... would find the same lines again, backwards.)
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = tuple((r + i * dr, c + i * dc) for i in range(length))   # the cells we walk on
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    lines.append(cells)                 # keep it only if it stays on the board
    return tuple(lines)                                 # a tuple cannot be changed by mistake later


# LINES[3] and LINES[4]: every way to win, for 3 in a row and for 4 in a row.
# You may read it (for example in an evaluation function); never change it.
LINES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {3: _build_lines(3), 4: _build_lines(4)}

# The same lines again, split into two numpy tables of numbers:
#   _LINE_ROWS[k] holds the row of every cell of every line, _LINE_COLS[k] the column.
#   Example: the first line ((0, 0), (0, 1), (0, 2)) is rows [0, 0, 0] and columns [0, 1, 2].
# has_line uses them to check all the lines at once, which is much faster than a loop.
_LINE_ROWS = {k: np.array([[r for r, c in ln] for ln in v], dtype=np.intp) for k, v in LINES.items()}
_LINE_COLS = {k: np.array([[c for r, c in ln] for ln in v], dtype=np.intp) for k, v in LINES.items()}



def _as_cell(value, name: str) -> tuple[int, int]:
    """Check one cell of a Move and clean it up. Move uses it for src and dest.

    Takes: value - the cell, e.g. (1, 2), or (np.int64(1), 2) when it comes from numpy.
           name  - "src" or "dest"; only used to make the error message clear.
    Gives: the cell as two plain Python whole numbers, e.g. (1, 2).
    Raises ValueError if value is not two whole numbers, or if the cell is off the board.
    """
    try:
        r, c = value                              # must be exactly two things
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (row, col) pair, got {value!r}")
    for x in (r, c):
        # whole numbers only: 1.5 or "1" are refused, and so are True / False
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            raise ValueError(f"{name} coordinates must be whole numbers, got {value!r}")
    r, c = int(r), int(c)                         # numpy numbers become plain Python numbers
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError(f"{name} {value!r} is off the board")
    return (r, c)




@dataclass(frozen=True, slots=True)
class Move:
    """One action in the game: put a new piece on the board, or move one that is already there.

    src  - None for a new piece from the reserve; otherwise the (row, col) the piece is lifted from
    dest - the (row, col) the piece goes to
    size - SMALL, MEDIUM or LARGE (0, 1 or 2)
    Example: Move(None, (1, 2), LARGE) puts a new large piece on row 1, column 2.
    """
    src: tuple[int, int] | None
    dest: tuple[int, int]
    size: int

    def __post_init__(self):
        """Runs by itself right after Move(...) is made. Checks the three values and cleans them up.

        src must be None (a placement) or a cell on the board; dest must be a cell on the board;
        size must be 0, 1 or 2. numpy numbers become plain Python numbers. Anything else
        raises ValueError. src == dest is allowed here; check_move_validity refuses it later.
        """
        src = None if self.src is None else _as_cell(self.src, "src")
        dest = _as_cell(self.dest, "dest")
        size = self.size
        # a whole number only: 1.5 or "L" are refused, and so are True / False
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise ValueError(f"size must be 0, 1 or 2, got {size!r}")
        size = int(size)
        if not (0 <= size < NUM_SIZES):
            raise ValueError(f"size must be 0, 1 or 2, got {size!r}")
        # The Move is frozen (read-only), so `self.src = src` would be refused.
        # object.__setattr__ is the way to store the cleaned values while the Move is being made.
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "size", size)

    def is_placement(self) -> bool:
        """True if this move puts a new piece from the reserve on the board (src is None)."""
        return self.src is None

    @classmethod
    def _unchecked(cls, src, dest, size) -> "Move":
        """Make a Move fast, skipping the checks of __post_init__.

        Only get_valid_moves uses it: it makes 50 to 80 moves per call from values it already
        knows are correct, and skipping the checks makes it about three times faster.
        """
        move = object.__new__(cls)
        object.__setattr__(move, "src", src)
        object.__setattr__(move, "dest", dest)
        object.__setattr__(move, "size", size)
        return move



def get_top(board: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Which piece is on top of cell (r, c)? The top piece is the biggest one in the cell.

    Gives (owner, size), e.g. (1, MEDIUM) when player 1's medium is on top.
    An empty cell gives (EMPTY, -1); the obstacle gives (OBSTACLE, -1).
    """
    stack = board[r, c]                           # the 3 slots of this cell: small, medium, large
    if stack[0] == OBSTACLE:                      # the obstacle fills all 3 slots, so slot 0 is enough
        return (OBSTACLE, -1)
    for s in (LARGE, MEDIUM, SMALL):              # look from the biggest down:
        if stack[s] != EMPTY:                     #   the first piece found is the one on top
            return (int(stack[s]), s)
    return (EMPTY, -1)



def top_view(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What you see when you look down at the board: the top piece of all 16 cells at once.

    Gives two 4x4 arrays:
      owners - who owns the top piece: 0 empty, 1 or 2 a player, 3 the obstacle
      sizes  - the size of the top piece, or -1 where the cell is empty or the obstacle
    Same answer as calling get_top on every cell, but much faster.
    """
    owners = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    sizes = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    for s in (SMALL, MEDIUM, LARGE):              # small first, so a bigger piece writes over a smaller one
        layer = board[:, :, s]                    # the slot of this size, on all 16 cells
        mask = (layer == PLAYER_1) | (layer == PLAYER_2)   # True where a piece of this size sits
        owners[mask] = layer[mask]
        sizes[mask] = s
    owners[board[:, :, 0] == OBSTACLE] = OBSTACLE   # last, mark the obstacle cell
    return owners, sizes


_CELLS = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]   # the 16 cells, made once


def get_valid_moves(board: np.ndarray, reserves: np.ndarray, player: int,
                    allow_moves: bool | None = None) -> list[Move]:
    """Every legal move for `player` right now, as a list of Move.

    The order is always the same: first all placements (small, then medium, then large),
    then all board moves. Board moves are left out when the rule turns them off.
    """
    owners, sizes = top_view(board)
    own, sz = owners.tolist(), sizes.tolist()         # plain Python lists: faster to read in loops
    moves: list[Move] = []
    for s in (SMALL, MEDIUM, LARGE):                  # placements
        if reserves[player, s] < 1:                   # no piece of this size left
            continue
        for (r, c) in _CELLS:
            if own[r][c] != OBSTACLE and sz[r][c] < s:   # not the obstacle, top piece smaller
                moves.append(Move._unchecked(None, (r, c), s))
    if not _resolve_allow_moves(allow_moves):         # board moves are off: done
        return moves
    # Board moves: from every cell where my piece is on top, to every other cell where it fits.
    for (r, c) in _CELLS:
        if own[r][c] != player:
            continue
        s = sz[r][c]
        for (r2, c2) in _CELLS:
            if (r2, c2) != (r, c) and own[r2][c2] != OBSTACLE and sz[r2][c2] < s:
                moves.append(Move._unchecked((r, c), (r2, c2), s))
    return moves


def _placement_ok(board, reserves, player, move) -> bool:
    """Can `player` put a new piece of size move.size from the reserve on move.dest?

    Yes only if all three are true: the player still has a piece of that size in the reserve,
    dest is not the obstacle, and the top piece at dest (if there is one) is smaller.
    The same size is not enough: a piece can only cover a smaller one.
    """
    if reserves[player, move.size] < 1:           # no piece of this size left
        return False
    owner, size = get_top(board, *move.dest)
    if owner == OBSTACLE:
        return False
    return size < move.size                       # an empty cell has size -1, so any piece fits


def _board_move_ok(board, player, move) -> bool:
    """Can `player` lift their piece from move.src and put it on move.dest?

    Yes only if all are true: dest is another cell; the top piece at src belongs to the player
    and has size move.size; dest is not the obstacle; and the top piece at dest (if there is
    one) is smaller. The same size is not enough: a piece can only cover a smaller one.
    """
    if move.src == move.dest:                     # the piece must go to another cell
        return False
    owner, size = get_top(board, *move.src)
    # The player's own piece, on top, of the size the move says. This one test also refuses
    # an empty cell, the obstacle, the opponent's piece, and a piece that is covered.
    if owner != player or size != move.size:
        return False
    dest_owner, dest_size = get_top(board, *move.dest)
    if dest_owner == OBSTACLE:
        return False
    return dest_size < move.size



def check_move_validity(board: np.ndarray, reserves: np.ndarray, player: int, move,
                        allow_moves: bool | None = None) -> bool:
    """Is this move legal for `player` right now? Gives True or False, and never an error.

    Takes: move        - normally a Move; anything else (None, a string, a tuple, ...) gives False
           allow_moves - True, False, or None for the rule of this game. When board moves are
                         off, every move that has a src gives False.
    """
    if not isinstance(move, Move):                # junk gives False, not an error
        return False
    if player not in (PLAYER_1, PLAYER_2):
        return False
    if move.is_placement():                       # placements are allowed in every game
        return _placement_ok(board, reserves, player, move)
    if not _resolve_allow_moves(allow_moves):     # board moves only when the rule allows them
        return False
    return _board_move_ok(board, player, move)



def execute_move(board: np.ndarray, reserves: np.ndarray, player: int, move: Move,
                 allow_moves: bool | None = None) -> None:
    """Play `move` for `player`. Changes `board` and `reserves` directly; gives nothing back.

    A placement takes one piece out of the reserve. A board move lifts the piece from src,
    so whatever was under it can be seen again.
    Raises ValueError if the move is illegal, and then nothing is changed.
    """
    # Check first: an illegal move must not change anything.
    if not check_move_validity(board, reserves, player, move, allow_moves):
        raise ValueError(f"illegal move for player {player}: {move!r}")
    if move.is_placement():
        reserves[player, move.size] -= 1          # one piece fewer in the reserve
    else:
        r, c = move.src
        board[r, c, move.size] = EMPTY            # empty only this size's slot: smaller pieces under it stay
    r, c = move.dest
    board[r, c, move.size] = player               # the piece goes into the slot of its size at dest



def check_winner(board: np.ndarray, last_mover: int, win_length: int | None = None) -> int:
    """Has someone won, right after a move? Gives the winner (1 or 2), or 0 if nobody has.

    Takes: last_mover - the player who just moved; win_length - 3, 4, or None.
    The OTHER player is checked first. If the mover lifted a piece and so uncovered the other
    player's line, the other player wins, even if the mover made a line with the same move.
    """
    owners, _ = top_view(board)                   # only the top pieces count
    opponent = 3 - last_mover                     # 3 - 1 = 2 and 3 - 2 = 1
    # The other player first: if they have a line now, this move uncovered it, so they win.
    if has_line(owners, opponent, win_length):
        return opponent
    if has_line(owners, last_mover, win_length):
        return last_mover
    return 0


def has_line(owners: np.ndarray, player: int, win_length: int | None = None) -> bool:
    """Does `player` own all the cells of at least one winning line? Gives True or False.

    Takes: owners     - the 4x4 owners array from top_view
           player     - 1 or 2
           win_length - 3, 4, or None for the rule of this game
    Example: row 0 of owners is [1, 1, 1, 0] -> True for player 1 with 3 in a row,
             but False with 4 in a row.
    """
    k = _resolve_win_length(win_length)
    # One row per line, one column per cell of that line: True where `player` owns the cell.
    hits = owners[_LINE_ROWS[k], _LINE_COLS[k]] == player
    return bool(hits.all(axis=1).any())          # is there a line that is True in every cell?


def check_endgame(board: np.ndarray, reserves: np.ndarray, player_to_move: int, ply: int,
                  allow_moves: bool | None = None, win_length: int | None = None) -> tuple[bool, int, str]:
    """Is the game over, just before `player_to_move` plays ply number `ply`?

    Gives (is_over, winner, reason). winner is 1 or 2, or 0 for a draw or a game that goes on.
    reason is "" while the game goes on. The checks run in this order:
      1. Someone has a line: that player wins. reason "line", or "revealed" when the last
         mover lifted a piece and so uncovered the other player's line.
      2. ply >= MAX_PLIES: a draw, reason "ply_cap".
      3. Board moves on:  player_to_move has no legal move at all: they lose, "no_moves".
         Board moves off: neither player can place a piece: a draw, "exhausted".
         (Only one stuck player is not the end: that player passes, see must_pass.)
    """
    am = _resolve_allow_moves(allow_moves)
    last_mover = 3 - player_to_move               # the player who played the ply before
    winner = check_winner(board, last_mover, win_length)
    if winner != 0:                               # 1. a line
        return (True, winner, "revealed" if winner == player_to_move else "line")
    if ply >= MAX_PLIES:                          # 2. too many plies
        return (True, 0, "ply_cap")
    mine = get_valid_moves(board, reserves, player_to_move, am)   # 3. any legal move left?
    if am:
        if not mine:
            return (True, last_mover, "no_moves")
        return (False, 0, "")
    if not mine and not get_valid_moves(board, reserves, last_mover, am):
        return (True, 0, "exhausted")
    return (False, 0, "")


def must_pass(board: np.ndarray, reserves: np.ndarray, player: int,
              allow_moves: bool | None = None) -> bool:
    """Must `player` skip this turn? Gives True or False.

    True only when board moves are off, nobody has won yet, the player has no legal placement
    left, and the opponent still has one. With board moves on it is always False.
    (If neither player can place, it is not a pass: the game ends in a draw, see check_endgame.)
    """
    am = _resolve_allow_moves(allow_moves)
    if am:                                        # moves on: never a pass (no move at all = a loss)
        return False
    if check_winner(board, 3 - player) != 0:      # the game is already won: nobody passes
        return False
    if get_valid_moves(board, reserves, player, am):   # the player can still play
        return False
    return bool(get_valid_moves(board, reserves, 3 - player, am))   # a pass only if the opponent can play



def random_move(board: np.ndarray, reserves: np.ndarray, player: int, rng=None,
                allow_moves: bool | None = None):
    """A random legal move for `player` (every legal move equally likely), or None if there is none.

    rng: a numpy random generator, e.g. np.random.default_rng(7). Generators made with the
    same seed make the same choices, so a game can be replayed. None makes a new unseeded one.
    This is the brain for the random agent.
    """
    moves = get_valid_moves(board, reserves, player, allow_moves)
    if not moves:
        return None
    if rng is None:
        rng = np.random.default_rng()
    return moves[int(rng.integers(len(moves)))]



def copy_state(board: np.ndarray, reserves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """New copies of board and reserves. Changing a copy does not change the original.

    Why you need it: execute_move changes the board you give it, and `b = board` is NOT a
    copy (both names point to the same board). To try a move without really playing it:

        b, r = copy_state(board, reserves)        # make copies
        execute_move(b, r, player, move)          # play the move on the copies only
        if check_winner(b, player) == player:     # look at the result
            ...
        # board and reserves are still exactly as before

    Use a fresh copy for every move you try (for example in a search).
    """
    return board.copy(), reserves.copy()



def state_key(board: np.ndarray, reserves: np.ndarray, player: int) -> bytes:
    """One bytes value that stands for the whole position: board, reserves and player to move.

    Two positions give the same value only if they are exactly the same, so it works as a
    dictionary key, e.g. to remember positions your search has already looked at.
    """
    return board.tobytes() + reserves.tobytes() + bytes([player])


def new_game(seed: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    """Make the start of a game from its seed. The same seed always gives the same start.

    Takes: seed - a whole number, 0 or bigger (numpy refuses a negative seed with ValueError).
    Gives four things:
      board        - 4x4x3 array, empty except the obstacle cell (all 3 of its slots are OBSTACLE)
      reserves     - 3x3 array: reserves[p, s] = how many pieces of size s player p has left
                     (row 0 is not used). Both players get 3/3/3 minus one piece of the same size.
      obstacle     - (row, col) of the obstacle cell
      removed_size - the size that was taken away: SMALL, MEDIUM or LARGE
    """
    rng = np.random.default_rng(seed)
    cell = int(rng.integers(BOARD_SIZE * BOARD_SIZE))    # first draw the obstacle cell (0 to 15) ...
    removed = int(rng.integers(NUM_SIZES))               # ... then the removed size (this order sets each seed's game)
    board = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_SIZES), dtype=np.int8)
    obstacle = (cell // BOARD_SIZE, cell % BOARD_SIZE)   # cell number -> (row, col), e.g. 11 -> (2, 3)
    board[obstacle[0], obstacle[1], :] = OBSTACLE
    reserves = np.zeros((3, NUM_SIZES), dtype=np.int8)
    for p in (PLAYER_1, PLAYER_2):
        reserves[p, :] = INITIAL_RESERVE                 # 3 small, 3 medium, 3 large ...
        reserves[p, removed] -= 1                        # ... minus one piece of the removed size
    return board, reserves, obstacle, removed




def render_text(board: np.ndarray, reserves: np.ndarray) -> str:
    """Draw the board and both reserves as text, for logs and for the human agent.

    Each cell lists its pieces from bottom to top: "2S<1M" = player 2's small under player 1's
    medium. "." is an empty cell and "##" the obstacle. Gives one string with several lines.
    """
    lines = []
    for r in range(BOARD_SIZE):
        cells = []
        for c in range(BOARD_SIZE):
            stack = board[r, c]
            if stack[0] == OBSTACLE:
                text = "##"
            else:
                # one "owner + size" text per piece, bottom to top, e.g. ["2S", "1M"]
                parts = [f"{int(stack[s])}{SIZE_NAMES[s]}" for s in range(NUM_SIZES) if stack[s] != EMPTY]
                text = "<".join(parts) if parts else "."
            cells.append(f"{text:>8}")                # every cell 8 characters wide, so columns line up
        lines.append(f"{r} " + "".join(cells))
    lines.append("    " + "".join(f"{c:>8}" for c in range(BOARD_SIZE)))
    for p in (PLAYER_1, PLAYER_2):
        counts = " ".join(f"{SIZE_NAMES[s]}:{int(reserves[p, s])}" for s in range(NUM_SIZES))
        lines.append(f"reserve P{p}: {counts}")
    return "\n".join(lines)