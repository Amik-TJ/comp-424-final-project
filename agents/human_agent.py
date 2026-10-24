"""An agent that lets you play from the keyboard.

Type `p S 1 2` to place a small piece on row 1, column 2, `m 0 0 1 2` to move your top piece
from (0, 0) to (1, 2), or `q` to quit. Sizes can be written S, M, L or 0, 1, 2.
"""
from __future__ import annotations

import sys

import constants
from agents.agent import Agent
from helpers import Move, check_move_validity, get_valid_moves, render_text, get_top
from store import register_agent

SIZE_LETTERS = {"S": 0, "M": 1, "L": 2, "0": 0, "1": 1, "2": 2}   # what the player may type for a size


def _numbers(words):
    """Return typed rows and columns as integers, e.g. ["1", "2"] -> [1, 2].

    Raises ValueError with a message for the player if one of them is not a number.
    """
    try:
        return [int(w) for w in words]
    except ValueError:
        raise ValueError("rows and columns must be numbers, in this order: `p S r c` "
                         "(e.g. `p S 1 2`) or `m r c r2 c2`") from None


def parse_human_move(text: str):
    """Return the Move that one line of keyboard input describes.

    Takes:   text - one line, e.g. "p S 1 2" (place a small piece on row 1, column 2)
                    or "m 0 0 1 2" (move the top piece from (0, 0) to (1, 2)).
    Returns: a Move. For "m" the size is a placeholder 0; HumanAgent.step replaces it with the
             size of the lifted piece.
    Raises ValueError, with a message for the player, when the line cannot be read.
    """
    parts = text.strip().split()
    if not parts:
        raise ValueError("empty line")
    kind = parts[0].lower()
    if kind == "p" and len(parts) == 4:
        size = SIZE_LETTERS.get(parts[1].upper())
        if size is None:
            raise ValueError("size must be S, M or L")
        r, c = _numbers(parts[2:])
        return Move(None, (r, c), size)
    if kind == "m" and len(parts) == 5:
        if not constants.ALLOW_MOVES:
            raise ValueError("board moves are disabled in this game")
        r, c, r2, c2 = _numbers(parts[1:])
        return Move((r, c), (r2, c2), 0)
    raise ValueError("use `p S r c` to place or `m r c r2 c2` to move")


def move_to_text(move) -> str:
    """Return the text to type for `move`; the opposite of parse_human_move.

    Examples: Move(None, (1, 2), SMALL) -> "p S 1 2";  Move((0, 0), (1, 2), LARGE) -> "m 0 0 1 2"
    """
    if move.src is None:
        return f"p {constants.SIZE_NAMES[move.size]} {move.dest[0]} {move.dest[1]}"
    return f"m {move.src[0]} {move.src[1]} {move.dest[0]} {move.dest[1]}"


def help_text() -> str:
    """Return the instructions for typing a move, with an example of each kind of input.

    The board-move line is left out when board moves are off in this game.
    """
    rows = [("p S r c", "place a piece from your reserve: size S, M or L, then row and column", "p S 1 2")]
    if constants.ALLOW_MOVES:
        rows.append(("m r c r2 c2", "move your top piece from row r, column c to row r2, column c2", "m 0 0 1 2"))
    width = max(len(meaning) for _, meaning, _ in rows)    # so the examples line up
    lines = ["How to type a move (row 0 is the top row, column 0 is the left column):"]
    lines += [f"  {form:<13} {meaning:<{width}}   Example: {example}" for form, meaning, example in rows]
    lines.append(f"  {'q':<13} quit the game")
    return "\n".join(lines)


@register_agent("human_agent")
class HumanAgent(Agent):
    """Asks the person at the keyboard for every move."""

    def __init__(self):
        """Create the agent. autoplay is False: the agent needs a person at the keyboard, and the
        referee never gives it a strike for taking time to think.
        """
        super().__init__()
        self.name = "HumanAgent"
        self.autoplay = False
        self.help_shown = False             # the typing instructions are printed before the first move

    def step(self, board, reserves, player, opponent):
        """Print the board and ask for a move until the player types a legal one.

        Takes the same arguments as Agent.step. Returns the legal Move that was typed.
        Typing `q`, Ctrl-D or Ctrl-C ends the program (SystemExit).
        """
        print(render_text(board, reserves))
        if not self.help_shown:
            print(help_text())
            self.help_shown = True
        legal = get_valid_moves(board, reserves, player)
        example = move_to_text(legal[0]) if legal else "p S 1 2"     # a move that is legal right now
        prompt = f"Your move as {constants.PLAYER_NAMES[player]} (player {player}), e.g. {example}  (q to quit): "
        while True:
            try:
                text = input(prompt)
            except (EOFError, KeyboardInterrupt):   # Ctrl-D or Ctrl-C: the same as typing q
                print()
                text = "q"
            if text.strip().lower() == "q":
                print("Game ended by user!")
                sys.exit(0)
            try:
                move = parse_human_move(text)
                if move.src is not None:            # put in the real size of the piece being lifted
                    _, size = get_top(board, *move.src)
                    move = Move(move.src, move.dest, max(size, 0))
            except ValueError as exc:
                print(f"Bad input: {exc}")
                print(help_text())
                continue
            if check_move_validity(board, reserves, player, move):
                return move
            print("Illegal move. You can type one of these:")
            texts = [move_to_text(m) for m in legal]
            for i in range(0, len(texts), 10):      # ten moves per line, never split in the middle
                print("  " + ", ".join(texts[i:i + 10]))
