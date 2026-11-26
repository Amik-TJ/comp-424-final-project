"""Compact game-state representation used by the search algorithm.

The mixin owns the internal board representation, incremental hash/bitboards,
move application and undo, terminal checks, and tactical threat detection.
"""

import constants
from helpers import LINES, get_valid_moves

from agents import board_storage
from agents.move_generation import generate_custom_moves, generate_original_moves

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
MAX_TT_ENTRIES = 250_000
COORDS = tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))


class GameStateMixin:
    """Methods that maintain and query the compact position stored on an agent."""

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
                (s0 << (base * board_storage.BITS_PER_ENTRY))
                | (s1 << ((base + 1) * board_storage.BITS_PER_ENTRY))
                | (s2 << ((base + 2) * board_storage.BITS_PER_ENTRY))
            )
        for p in (PLAYER_1, PLAYER_2):
            for s in PIECE_SIZES:
                self._hash_write_res(p, s)
        if player == PLAYER_2:
            self.z |= board_storage.PLAYER_BIT
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
        self._reset_move_ordering()
        self.nodes = 0

    def _hash_write_slot(self, idx, val):
        board_storage.write_slot_hash(self, idx, val)

    def _hash_write_res(self, player, size):
        board_storage.write_reserve_hash(self, player, size)

    def _set_top(self, i, owner, size):
        board_storage.set_top_piece(self, i, owner, size)

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
        self.z ^= board_storage.PLAYER_BIT
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
        self.z ^= board_storage.PLAYER_BIT

    def _gen_moves(self, player):
        if self.use_custom_move_generation:
            return generate_custom_moves(self, player)
        return generate_original_moves(
            self, player, get_valid_moves, COORDS
        )

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
