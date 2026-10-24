"""Draws the board in a window with matplotlib. Used by `simulator.py --display`."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from constants import BOARD_SIZE, NUM_SIZES, OBSTACLE, EMPTY, SIZE_NAMES, PLAYER_NAMES

COLORS = {1: "tab:blue", 2: "tab:brown"}   # the color of each player's pieces
RADII = (0.2, 0.3, 0.42)                   # the radius of a SMALL, MEDIUM and LARGE piece, in cells
SAVE_DIR = None        # a folder: render() also saves every frame there (simulator.py --display_save)


class UIEngine:
    """Draws one game in a matplotlib window, once after every ply."""

    def __init__(self, world):
        """Open the window.

        Takes: world - the World being played; render() reads its rules, ply, player, seed and
               display_delay.
        """
        self.world = world
        self.step_number = 0
        plt.figure(figsize=(7, 6))
        plt.ion()                                    # interactive mode: drawing does not wait for the window

    def _draw_stack(self, ax, r, c, stack):
        """Draw the pieces of one cell.

        Takes: ax    - the drawing area.
               r, c  - the cell's row and column.
               stack - board[r, c], the three size slots of that cell.
        Each piece is a ring of its own size, so pieces underneath stay visible; the top piece
        is also filled in. The obstacle is a dark gray square.
        """
        x, y = c + 0.5, (BOARD_SIZE - 1 - r) + 0.5      # row 0 is drawn at the top
        if stack[0] == OBSTACLE:
            ax.add_patch(plt.Rectangle((c + 0.05, BOARD_SIZE - 1 - r + 0.05), 0.9, 0.9, color="dimgray"))
            return
        for s in range(NUM_SIZES - 1, -1, -1):            # draw largest first so small rings stay visible
            if stack[s] != EMPTY:
                ax.add_patch(plt.Circle((x, y), RADII[s], color=COLORS[int(stack[s])], fill=False, lw=3))
        for s in range(NUM_SIZES - 1, -1, -1):            # fill the top piece: the largest one in the cell
            if stack[s] != EMPTY:
                ax.add_patch(plt.Circle((x, y), RADII[s] - 0.03, color=COLORS[int(stack[s])], alpha=0.35))
                break

    def render(self, board, reserves, record):
        """Draw the whole board and the text under it, then pause for display_delay seconds.

        Takes: board, reserves - the position to draw.
               record          - the game's GameRecord, for the last move and the result.
        The text shows the rules, both reserves, whose turn it is (or the result), and the last
        move. With SAVE_DIR set, the frame is also saved as a PNG file.
        """
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, BOARD_SIZE); ax.set_ylim(0, BOARD_SIZE); ax.set_aspect("equal")
        for i in range(BOARD_SIZE + 1):
            ax.plot([0, BOARD_SIZE], [i, i], color="silver"); ax.plot([i, i], [0, BOARD_SIZE], color="silver")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self._draw_stack(ax, r, c, board[r, c])
        ax.set_xticks([i + 0.5 for i in range(BOARD_SIZE)]); ax.set_xticklabels(range(BOARD_SIZE))
        ax.set_yticks([i + 0.5 for i in range(BOARD_SIZE)]); ax.set_yticklabels(reversed(range(BOARD_SIZE)))
        ax.xaxis.tick_top()
        w = self.world
        rules = f"moves: {'on' if w.rules['allow_moves'] else 'off'}  win: {w.rules['win_length']}"
        res = "   ".join(f"{PLAYER_NAMES[p]} " + " ".join(f"{SIZE_NAMES[s]}:{int(reserves[p, s])}" for s in range(NUM_SIZES)) for p in (1, 2))
        status = f"ply {w.ply}  to move: {PLAYER_NAMES[w.player]}"
        if record.winner is not None:
            status = "draw" if record.winner == 0 else f"{PLAYER_NAMES[record.winner]} wins ({record.reason})"
        if record.moves:                             # the last ply, e.g. "last: Brown passed" or "last: Blue L (0, 1) -> (2, 2)"
            m = record.moves[-1]
            who = PLAYER_NAMES[m["player"]]
            if m.get("pass"):
                status += f"   last: {who} passed"
            else:
                src = "" if m["src"] is None else f"{tuple(m['src'])} -> "
                status += f"   last: {who} {SIZE_NAMES[m['size']]} {src}{tuple(m['dest'])}"
        plt.figtext(0.5, 0.06, f"{rules}\nreserves  {res}\n{status}", ha="center")
        plt.subplots_adjust(bottom=0.22)
        plt.pause(max(w.display_delay, 0.01))
        if SAVE_DIR is not None:                     # --display_save: one image per frame, e.g. plots/seed7_003.png
            os.makedirs(SAVE_DIR, exist_ok=True)
            plt.savefig(os.path.join(SAVE_DIR, f"seed{w.seed}_{self.step_number:03d}.png"))
        self.step_number += 1
