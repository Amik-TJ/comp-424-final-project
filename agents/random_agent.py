"""An agent that plays a random legal move. Use it as the first opponent for your own agent."""
from __future__ import annotations

import numpy as np

from agents.agent import Agent
from helpers import random_move
from store import register_agent


@register_agent("random_agent")
class RandomAgent(Agent):
    """Picks one of the legal moves at random. Every legal move is equally likely."""

    def __init__(self):
        """Create the agent with its own random number generator."""
        super().__init__()
        self.name = "RandomAgent"
        self.rng = np.random.default_rng()

    def step(self, board, reserves, player, opponent):
        """Return a random legal Move for `player`.

        Takes the same arguments as Agent.step: the board, the reserves, the agent's player
        number, and the opponent's number.
        """
        return random_move(board, reserves, player, self.rng)
