"""Base class every agent extends """
from __future__ import annotations

class Agent:
    def __init__(self):
        self.name = "DummyAgent"
        # True for every agent that plays by itself. Only the human agent sets it to False,
        # because it needs a person at the keyboard. When it is False:
        #   - the simulator refuses to use this agent with --autoplay (it would wait for input forever);
        #   - if step() raises SystemExit (the person typed q), the program stops instead of
        #     counting a strike.
        self.autoplay = True

    def __str__(self) -> str:
        return str(self.name)

    def step(self, board, reserves, player, opponent):
        """
        board    : np.ndarray of shape (4, 4, 3), int8: 4 rows x 4 columns x 3 size slots (S, M, L) per cell.
                This is your own copy of the current board. Changing it does not change the real game,
                so you can use it as scratch space for your search. You get a fresh copy every turn.
                Your returned move is checked against the real board, not against your copy.
        reserves : np.ndarray of shape (3, 3), int8: reserves[player, size] = how many pieces of that size
           `player` has not placed yet. Row 0 is unused (always 0), so row 1 is player 1 and
           row 2 is player 2. Also your own copy, with the same rules as board.
        Example:
        reserves == [
                    [0, 0, 0],    # row 0: unused
                    [3, 3, 2],    # player 1: 3 small, 3 medium, 2 large
                    [3, 3, 0]     # player 2: 3 small, 3 medium, 0 large 
                    ]    
        reserves[player_1, MEDIUM]   # how many mediums `player_1` still has

        player   : your player number, 1 or 2.
        opponent : the other player's number, 2 or 1.
        Returns a helpers.Move: the move you want to play, e.g. Move(None, (1, 2), LARGE) to place a
        large piece from your reserve on row 1, column 2. Picking from helpers.get_valid_moves() is always legal.
        You must return within STEP_TIME_LIMIT seconds (2.0, see constants.py), measured with
        time.perf_counter(); stop your search early and keep a safety margin. A late, illegal or missing
        move costs a strike and is replaced by a random legal move. 4 strikes lose the game.

        """
        raise NotImplementedError
