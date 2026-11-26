"""Low-level packed-board/hash/bitboard storage helpers."""

BITS_PER_ENTRY = 2
ENTRY_MASK = (1 << BITS_PER_ENTRY) - 1
NUM_SLOTS = 4 * 4 * 3
NUM_SIZES = 3
RES_BIT_OFFSET = NUM_SLOTS * BITS_PER_ENTRY
PLAYER_BIT_OFFSET = RES_BIT_OFFSET + 2 * NUM_SIZES * BITS_PER_ENTRY
PLAYER_BIT = 1 << PLAYER_BIT_OFFSET

def write_slot_hash(state, idx, value):
    shift = idx * BITS_PER_ENTRY
    state.z = (state.z & ~(ENTRY_MASK << shift)) | (value << shift)

def write_reserve_hash(state, player, size):
    shift = RES_BIT_OFFSET + ((player - 1) * NUM_SIZES + size) * BITS_PER_ENTRY
    state.z = (state.z & ~(ENTRY_MASK << shift)) | (state.res[player][size] << shift)

def set_top_piece(state, cell, owner, size):
    old_owner = state.own[cell]
    if old_owner != owner:
        bit = 1 << cell
        state.bb[old_owner] &= ~bit
        state.bb[owner] |= bit
        state.own[cell] = owner
    state.sz[cell] = size
