"""The base class that every agent extends."""
from __future__ import annotations


class Agent:
    """The base class of every agent. A subclass sets its name in __init__ and overrides step()."""

    def __init__(self):
        """Set the default name and the autoplay flag. Subclasses call it with super().__init__()."""
        self.name = "DummyAgent"
        # False only for an agent that needs a person at the keyboard (the human agent). The
        # simulator then refuses it with --autoplay, and SystemExit from its step() (the person
        # typed q) ends the program instead of costing a strike.
        self.autoplay = True

    def __str__(self) -> str:
        """Return the agent's name."""
        return str(self.name)

    def step(self, board, reserves, player, opponent):
        """Return the move to play in the current position. Every agent overrides this method.

        board    : np.ndarray of shape (4, 4, 3), int8: 4 rows x 4 columns x 3 size slots (S, M, L)
                   per cell. Your own copy, made fresh for every call: you may change it, e.g. as
                   scratch space for a search. The move you return is checked against the real board.
        reserves : np.ndarray of shape (3, 3), int8: reserves[player, size] is the number of pieces
                   of that size that `player` has not placed yet. Row 0 is unused (always 0), so
                   row 1 is player 1 and row 2 is player 2. Also your own copy.
                   Example:
                       reserves == [[0, 0, 0],    # row 0: unused
                                    [3, 3, 2],    # player 1: 3 small, 3 medium, 2 large
                                    [3, 3, 0]]    # player 2: 3 small, 3 medium, 0 large
                       reserves[player, MEDIUM]   # the number of mediums you have left
        player   : your player number, 1 or 2.
        opponent : the other player's number, 2 or 1.
        Returns: a helpers.Move, e.g. Move(None, (1, 2), LARGE) to place a large piece from your
                 reserve on row 1, column 2. Every move in helpers.get_valid_moves() is legal.
        Return within STEP_TIME_LIMIT seconds (2.0, see constants.py), measured with
        time.perf_counter(), and keep a safety margin. A late, illegal or missing move costs a
        strike and is replaced by a random legal move; the 4th strike loses the game.
        """
        raise NotImplementedError
