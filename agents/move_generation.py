"""Custom and framework-backed move generation for the search position."""

import numpy as np

PLACEMENT_SRC = -1
PIECE_SIZES = (0, 1, 2)
BOARD_SIZE = 4
NUM_SIZES = 3


def generate_custom_moves(state, player):
    sizes, owners = state.sz, state.own
    reserves, playable = state.res[player], state.playable
    moves = [
        (PLACEMENT_SRC, dest, size)
        for size in PIECE_SIZES
        if reserves[size] > 0
        for dest in playable
        if sizes[dest] < size
    ]
    if state.allow_moves:
        for src in playable:
            if owners[src] == player:
                size = sizes[src]
                moves.extend(
                    (src, dest, size)
                    for dest in playable
                    if dest != src and sizes[dest] < size
                )
    return moves


def generate_original_moves(state, player, get_valid_moves, coords):
    """Call the project's real helpers.get_valid_moves on this search position.

    The search stores its position in compact Python lists, whereas the framework
    helper expects NumPy arrays shaped (4, 4, 3) and (3, 3). Rebuild those arrays
    for the current node, call the real helper, then translate its Move objects
    (which use .src and .dest) into the search's integer-index representation.
    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_SIZES), dtype=np.int8)
    reserves = np.zeros((3, NUM_SIZES), dtype=np.int8)

    for index, (row, col) in enumerate(coords):
        base = index * NUM_SIZES
        board[row, col, :] = state.cells[base:base + NUM_SIZES]

    for reserve_player in (1, 2):
        reserves[reserve_player, :] = state.res[reserve_player]

    framework_moves = get_valid_moves(board, reserves, player)

    coord_to_index = {coord: i for i, coord in enumerate(coords)}
    packed = []
    for move in framework_moves:
        src = PLACEMENT_SRC if move.src is None else coord_to_index[tuple(move.src)]
        dest = coord_to_index[tuple(move.dest)]
        packed.append((src, dest, int(move.size)))
    return packed
