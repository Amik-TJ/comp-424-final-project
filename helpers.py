"""The rules of Goblet Gobblers, COMP 424 variant.

These functions only compute answers: nothing is printed or saved, and only execute_move changes
the arrays it is given. The rule flags constants.ALLOW_MOVES and constants.WIN_LENGTH are read
when allow_moves / win_length is left as None.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import constants
from constants import (BOARD_SIZE, NUM_SIZES, SMALL, MEDIUM, LARGE, EMPTY, PLAYER_1,
                       PLAYER_2, OBSTACLE, INITIAL_RESERVE, MAX_PLIES, SIZE_NAMES)


def _resolve_allow_moves(allow_moves: bool | None) -> bool:
    """Return whether board moves are allowed for this call.

    Takes: allow_moves - True, False, or None for the rule of the current game.
    None reads constants.ALLOW_MOVES at every call, because the framework sets the rule after
    this module is loaded.
    """
    if allow_moves is None:
        return bool(constants.ALLOW_MOVES)
    return bool(allow_moves)


def _resolve_win_length(win_length: int | None) -> int:
    """Return the number of pieces in a row that win for this call: 3 or 4.

    Takes: win_length - 3, 4, or None for the rule of the current game (constants.WIN_LENGTH,
                        read at every call).
    Raises ValueError for any other value.
    """
    if win_length is None:
        win_length = constants.WIN_LENGTH
    if win_length not in (3, 4):
        raise ValueError(f"win_length must be 3 or 4, got {win_length!r}")
    return int(win_length)


def _build_lines(length: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return every straight line of `length` cells on the board: rows, columns and diagonals.

    Takes:   length - 3 or 4.
    Returns: a tuple of lines, each a tuple of (row, col) cells, e.g. ((0, 0), (0, 1), (0, 2)):
             24 lines for 3, 10 lines for 4.
    Lines through the obstacle are kept: nobody owns the obstacle cell, so they can never win.
    """
    lines = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # Right, down and the two downward diagonals find every line exactly once.
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = tuple((r + i * dr, c + i * dc) for i in range(length))
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    lines.append(cells)
    return tuple(lines)


# LINES[3] and LINES[4]: every line of 3 and of 4 cells. Read it, e.g. in an evaluation
# function; never change it.
LINES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {3: _build_lines(3), 4: _build_lines(4)}

# The rows and the columns of the cells of every line, as index arrays, so that has_line checks
# all lines in one numpy operation.
_LINE_ROWS = {k: np.array([[r for r, c in ln] for ln in v], dtype=np.intp) for k, v in LINES.items()}
_LINE_COLS = {k: np.array([[c for r, c in ln] for ln in v], dtype=np.intp) for k, v in LINES.items()}


def _as_cell(value, name: str) -> tuple[int, int]:
    """Check one cell of a Move and return it as two Python ints.

    Takes:   value - the cell, e.g. (1, 2) or (np.int64(1), 2).
             name  - "src" or "dest", for the error message.
    Returns: the cell, e.g. (1, 2).
    Raises ValueError if value is not two integers or is off the board.
    """
    try:
        r, c = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (row, col) pair, got {value!r}")
    for x in (r, c):
        # bool is a subclass of int, so it is refused explicitly.
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            raise ValueError(f"{name} coordinates must be whole numbers, got {value!r}")
    r, c = int(r), int(c)
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError(f"{name} {value!r} is off the board")
    return (r, c)


@dataclass(frozen=True, slots=True)
class Move:
    """One action: place a piece from the reserve, or move a piece that is on the board.

    src  - None for a placement; otherwise the (row, col) the piece is lifted from
    dest - the (row, col) the piece goes to
    size - SMALL, MEDIUM or LARGE (0, 1 or 2)
    Example: Move(None, (1, 2), LARGE) places a large piece on row 1, column 2.
    """
    src: tuple[int, int] | None
    dest: tuple[int, int]
    size: int

    def __post_init__(self):
        """Check the three fields and convert numpy integers to Python ints.

        Raises ValueError if src is neither None nor a cell on the board, if dest is not a cell
        on the board, or if size is not 0, 1 or 2. src == dest is accepted here;
        check_move_validity refuses it.
        """
        src = None if self.src is None else _as_cell(self.src, "src")
        dest = _as_cell(self.dest, "dest")
        size = self.size
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise ValueError(f"size must be 0, 1 or 2, got {size!r}")
        size = int(size)
        if not (0 <= size < NUM_SIZES):
            raise ValueError(f"size must be 0, 1 or 2, got {size!r}")
        # The dataclass is frozen, so the converted values are stored with object.__setattr__.
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dest", dest)
        object.__setattr__(self, "size", size)

    def is_placement(self) -> bool:
        """Return True if this move places a piece from the reserve (src is None)."""
        return self.src is None

    @classmethod
    def _unchecked(cls, src, dest, size) -> "Move":
        """Return a Move built without the checks of __post_init__.

        Only get_valid_moves uses it, for values it has already checked; this makes it about
        three times faster.
        """
        move = object.__new__(cls)
        object.__setattr__(move, "src", src)
        object.__setattr__(move, "dest", dest)
        object.__setattr__(move, "size", size)
        return move


def get_top(board: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Return (owner, size) of the top piece of cell (r, c), the largest piece in the cell.

    Example: (1, MEDIUM) when player 1's medium is on top. An empty cell gives (EMPTY, -1),
    the obstacle (OBSTACLE, -1).
    """
    stack = board[r, c]
    if stack[0] == OBSTACLE:                      # the obstacle fills all three slots
        return (OBSTACLE, -1)
    for s in (LARGE, MEDIUM, SMALL):
        if stack[s] != EMPTY:
            return (int(stack[s]), s)
    return (EMPTY, -1)


def top_view(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the top piece of every cell as two 4 x 4 arrays, (owners, sizes).

    owners - the owner of each top piece: 0 empty, 1 or 2 a player, 3 the obstacle
    sizes  - the size of each top piece; -1 where the cell is empty or the obstacle
    The same answer as get_top on every cell, but much faster.
    """
    owners = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    sizes = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)
    for s in (SMALL, MEDIUM, LARGE):              # smallest first, so a larger piece overwrites it
        layer = board[:, :, s]
        mask = (layer == PLAYER_1) | (layer == PLAYER_2)
        owners[mask] = layer[mask]
        sizes[mask] = s
    owners[board[:, :, 0] == OBSTACLE] = OBSTACLE
    return owners, sizes


_CELLS = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]


def get_valid_moves(board: np.ndarray, reserves: np.ndarray, player: int,
                    allow_moves: bool | None = None) -> list[Move]:
    """Return every legal move of `player` as a list of Move.

    The order is fixed: placements first (small, then medium, then large), then board moves.
    Board moves are left out when the rule turns them off.
    """
    owners, sizes = top_view(board)
    own, sz = owners.tolist(), sizes.tolist()         # Python lists are faster to index in loops
    moves: list[Move] = []
    for s in (SMALL, MEDIUM, LARGE):
        if reserves[player, s] < 1:
            continue
        for (r, c) in _CELLS:
            if own[r][c] != OBSTACLE and sz[r][c] < s:
                moves.append(Move._unchecked(None, (r, c), s))
    if not _resolve_allow_moves(allow_moves):
        return moves
    # Board moves: each top piece of `player` to every other cell where it fits.
    for (r, c) in _CELLS:
        if own[r][c] != player:
            continue
        s = sz[r][c]
        for (r2, c2) in _CELLS:
            if (r2, c2) != (r, c) and own[r2][c2] != OBSTACLE and sz[r2][c2] < s:
                moves.append(Move._unchecked((r, c), (r2, c2), s))
    return moves


def _placement_ok(board, reserves, player, move) -> bool:
    """Return True if `player` can place a piece of size move.size on move.dest.

    The player must have a piece of that size in the reserve, and dest must not be the obstacle
    and must be empty or topped by a strictly smaller piece.
    """
    if reserves[player, move.size] < 1:
        return False
    owner, size = get_top(board, *move.dest)
    if owner == OBSTACLE:
        return False
    return size < move.size                       # an empty cell has size -1


def _board_move_ok(board, player, move) -> bool:
    """Return True if `player` can move the top piece at move.src to move.dest.

    dest must differ from src; the top piece at src must be the player's own and of size
    move.size; dest must not be the obstacle and must be empty or topped by a strictly smaller piece.
    """
    if move.src == move.dest:
        return False
    owner, size = get_top(board, *move.src)
    # This test also refuses an empty cell, the obstacle, an opponent's piece and a covered piece.
    if owner != player or size != move.size:
        return False
    dest_owner, dest_size = get_top(board, *move.dest)
    if dest_owner == OBSTACLE:
        return False
    return dest_size < move.size


def check_move_validity(board: np.ndarray, reserves: np.ndarray, player: int, move,
                        allow_moves: bool | None = None) -> bool:
    """Return True if `move` is legal for `player` now. Never raises.

    Takes: move        - a Move; any other value (None, a string, a tuple, ...) gives False.
           allow_moves - True, False, or None for the rule of the current game. With board
                         moves off, every move with a src gives False.
    """
    if not isinstance(move, Move):
        return False
    if player not in (PLAYER_1, PLAYER_2):
        return False
    if move.is_placement():
        return _placement_ok(board, reserves, player, move)
    if not _resolve_allow_moves(allow_moves):
        return False
    return _board_move_ok(board, player, move)


def execute_move(board: np.ndarray, reserves: np.ndarray, player: int, move: Move,
                 allow_moves: bool | None = None) -> None:
    """Play `move` for `player`, changing `board` and `reserves` in place.

    A placement takes the piece from the reserve. A board move lifts the piece from src, which
    uncovers the piece under it, if any.
    Raises ValueError if the move is illegal; nothing is changed then.
    """
    if not check_move_validity(board, reserves, player, move, allow_moves):
        raise ValueError(f"illegal move for player {player}: {move!r}")
    if move.is_placement():
        reserves[player, move.size] -= 1
    else:
        r, c = move.src
        board[r, c, move.size] = EMPTY            # only this size's slot: the pieces under it stay
    r, c = move.dest
    board[r, c, move.size] = player


def check_winner(board: np.ndarray, last_mover: int, win_length: int | None = None) -> int:
    """Return the winner right after `last_mover` moved: 1 or 2, or 0 if nobody has a line.

    Takes: last_mover - the player who just moved.
           win_length - 3, 4, or None for the rule of the current game.
    The opponent is checked first: if the move uncovered the opponent's line, the opponent wins,
    even if the mover completed a line with the same move.
    """
    owners, _ = top_view(board)                   # only top pieces count
    opponent = 3 - last_mover
    if has_line(owners, opponent, win_length):
        return opponent
    if has_line(owners, last_mover, win_length):
        return last_mover
    return 0


def has_line(owners: np.ndarray, player: int, win_length: int | None = None) -> bool:
    """Return True if `player` owns every cell of at least one line.

    Takes: owners     - the 4 x 4 owners array from top_view.
           player     - 1 or 2.
           win_length - 3, 4, or None for the rule of the current game.
    Example: row 0 of owners is [1, 1, 1, 0] -> True for player 1 with 3 in a row, False with 4.
    """
    k = _resolve_win_length(win_length)
    # One row per line, one column per cell of the line: True where `player` owns the cell.
    hits = owners[_LINE_ROWS[k], _LINE_COLS[k]] == player
    return bool(hits.all(axis=1).any())


def check_endgame(board: np.ndarray, reserves: np.ndarray, player_to_move: int, ply: int,
                  allow_moves: bool | None = None, win_length: int | None = None) -> tuple[bool, int, str]:
    """Return (is_over, winner, reason) just before `player_to_move` plays, after `ply` plies.

    winner is 1 or 2, or 0 for a draw or a game that goes on; reason is "" while the game goes
    on. The checks, in order:
      1. A player has a line: that player wins, with reason "line", or "revealed" when the last
         mover uncovered the opponent's line.
      2. ply >= MAX_PLIES: a draw, reason "ply_cap".
      3. Board moves on: player_to_move has no legal move and loses, reason "no_moves".
         Board moves off: neither player can place a piece, a draw, reason "exhausted".
         When only player_to_move is stuck, the game goes on and that player passes (must_pass).
    """
    am = _resolve_allow_moves(allow_moves)
    last_mover = 3 - player_to_move
    winner = check_winner(board, last_mover, win_length)
    if winner != 0:                               # 1. a line
        return (True, winner, "revealed" if winner == player_to_move else "line")
    if ply >= MAX_PLIES:                          # 2. the ply cap
        return (True, 0, "ply_cap")
    mine = get_valid_moves(board, reserves, player_to_move, am)   # 3. no legal move
    if am:
        if not mine:
            return (True, last_mover, "no_moves")
        return (False, 0, "")
    if not mine and not get_valid_moves(board, reserves, last_mover, am):
        return (True, 0, "exhausted")
    return (False, 0, "")


def must_pass(board: np.ndarray, reserves: np.ndarray, player: int,
              allow_moves: bool | None = None) -> bool:
    """Return True if `player` must skip this turn.

    Only with board moves off: nobody has won yet, the player has no legal placement, and the
    opponent still has one. If neither player can place, the game is a draw instead (check_endgame).
    """
    am = _resolve_allow_moves(allow_moves)
    if am:                                        # with moves on, a player with no move loses instead
        return False
    if check_winner(board, 3 - player) != 0:
        return False
    if get_valid_moves(board, reserves, player, am):
        return False
    return bool(get_valid_moves(board, reserves, 3 - player, am))


def random_move(board: np.ndarray, reserves: np.ndarray, player: int, rng=None,
                allow_moves: bool | None = None):
    """Return a random legal move of `player`, each equally likely, or None if there is none.

    Takes: rng - a numpy Generator, e.g. np.random.default_rng(7); the same seed gives the same
                 choices, so a game can be replayed. None makes a new, unseeded generator.
    """
    moves = get_valid_moves(board, reserves, player, allow_moves)
    if not moves:
        return None
    if rng is None:
        rng = np.random.default_rng()
    return moves[int(rng.integers(len(moves)))]


def copy_state(board: np.ndarray, reserves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of board and reserves. Changing a copy does not change the original.

    execute_move changes the arrays it is given, so try a move on copies:

        b, r = copy_state(board, reserves)
        execute_move(b, r, player, move)
        if check_winner(b, player) == player:
            ...

    Use fresh copies for every move a search tries.
    """
    return board.copy(), reserves.copy()


def state_key(board: np.ndarray, reserves: np.ndarray, player: int) -> bytes:
    """Return a bytes key for the position: board, reserves and player to move.

    Two positions have the same key only if they are identical, so the key can index a
    dictionary, e.g. a transposition table.
    """
    return board.tobytes() + reserves.tobytes() + bytes([player])


def new_game(seed: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    """Return the start of the game with this seed. The same seed always gives the same start.

    Takes:   seed - an integer, 0 or larger (numpy raises ValueError for a negative seed).
    Returns: (board, reserves, obstacle, removed_size):
             board        - 4 x 4 x 3, empty except the obstacle cell, whose three slots are OBSTACLE
             reserves     - 3 x 3, reserves[p, s] = the pieces of size s that player p has left
                            (row 0 unused); both players get 3/3/3 minus one piece of the same size
             obstacle     - (row, col) of the obstacle
             removed_size - the size removed: SMALL, MEDIUM or LARGE
    """
    rng = np.random.default_rng(seed)
    # The obstacle is drawn before the removed size; changing the order would change every seed's game.
    cell = int(rng.integers(BOARD_SIZE * BOARD_SIZE))
    removed = int(rng.integers(NUM_SIZES))
    board = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_SIZES), dtype=np.int8)
    obstacle = (cell // BOARD_SIZE, cell % BOARD_SIZE)   # cell number -> (row, col), e.g. 11 -> (2, 3)
    board[obstacle[0], obstacle[1], :] = OBSTACLE
    reserves = np.zeros((3, NUM_SIZES), dtype=np.int8)
    for p in (PLAYER_1, PLAYER_2):
        reserves[p, :] = INITIAL_RESERVE
        reserves[p, removed] -= 1
    return board, reserves, obstacle, removed


def render_text(board: np.ndarray, reserves: np.ndarray) -> str:
    """Return the board and both reserves as text, for logs and the human agent.

    Each cell lists its pieces from bottom to top, e.g. "2S<1M": player 2's small under player
    1's medium. "." is an empty cell and "##" the obstacle.
    """
    lines = []
    for r in range(BOARD_SIZE):
        cells = []
        for c in range(BOARD_SIZE):
            stack = board[r, c]
            if stack[0] == OBSTACLE:
                text = "##"
            else:
                parts = [f"{int(stack[s])}{SIZE_NAMES[s]}" for s in range(NUM_SIZES) if stack[s] != EMPTY]
                text = "<".join(parts) if parts else "."
            cells.append(f"{text:>8}")                # 8 characters per cell, so the columns line up
        lines.append(f"{r} " + "".join(cells))
    lines.append("    " + "".join(f"{c:>8}" for c in range(BOARD_SIZE)))
    for p in (PLAYER_1, PLAYER_2):
        counts = " ".join(f"{SIZE_NAMES[s]}:{int(reserves[p, s])}" for s in range(NUM_SIZES))
        lines.append(f"reserve P{p}: {counts}")
    return "\n".join(lines)
