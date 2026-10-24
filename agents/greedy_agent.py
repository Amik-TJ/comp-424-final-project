"""A one-ply agent: it wins at once if it can, never hands the opponent an immediate win, and
otherwise plays at random. It is a stronger opponent than random_agent, and an example of
how to look ahead with the helpers."""
from __future__ import annotations

import numpy as np

from agents.agent import Agent
from helpers import get_valid_moves, execute_move, check_winner, copy_state
from store import register_agent


def after(board, reserves, player, move):
    """Return the position after `player` plays `move`. The original arrays are not changed.

    Takes:   board, reserves - the position before the move.
             player          - the player who moves, 1 or 2.
             move            - a legal Move for that player.
    Returns: (board, reserves), new arrays with the move played.
    Example: b2, r2 = after(board, reserves, 1, Move(None, (0, 0), LARGE))
    """
    b, r = copy_state(board, reserves)
    execute_move(b, r, player, move)
    return b, r


@register_agent("greedy_agent")
class GreedyAgent(Agent):
    """Looks one move ahead: plays a winning move if there is one, and avoids moves after
    which the opponent can win at once."""

    def __init__(self):
        """Create the agent with its own random number generator."""
        super().__init__()
        self.name = "GreedyAgent"
        self.rng = np.random.default_rng()

    def step(self, board, reserves, player, opponent):
        """Return a Move for `player`, chosen in this order:

          1. a move that wins at once;
          2. a random move after which the opponent cannot win at once;
          3. if every move lets the opponent win, any random legal move.
        Takes the same arguments as Agent.step.
        """
        moves = get_valid_moves(board, reserves, player)
        safe = []                                   # moves after which the opponent cannot win at once
        for mv in moves:
            b, r = after(board, reserves, player, mv)
            w = check_winner(b, player)
            if w == player:
                return mv                           # this move wins the game now
            if w == opponent:
                continue                            # lifting this piece uncovers the opponent's line
            # Does the opponent have a reply that wins? any() stops at the first one it finds.
            opponent_wins = any(check_winner(after(b, r, opponent, om)[0], opponent) == opponent
                                for om in get_valid_moves(b, r, opponent))
            if not opponent_wins:
                safe.append(mv)
        pool = safe or moves                        # `safe` if it is not empty, else every move
        return pool[int(self.rng.integers(len(pool)))]
