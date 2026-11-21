"""A tactical baseline agent for Goblet Gobblers.

The agent first looks for one-move wins, then avoids moves that immediately
lose (including losses caused by uncovering an opponent's line), and finally
uses a small positional heuristic to choose among safe moves.
"""
from agents.agent import Agent
from store import register_agent
from helpers import (get_valid_moves, execute_move, check_winner, copy_state,
                     top_view, LINES)
from constants import *


def after(board, reserves, player, move):
    """Return copied state after ``player`` makes a legal ``move``."""
    next_board, next_reserves = copy_state(board, reserves)
    execute_move(next_board, next_reserves, player, move)
    return next_board, next_reserves


def opponent_has_immediate_win(board, reserves, opponent):
    """Return whether the opponent can win on their next move."""
    for move in get_valid_moves(board, reserves, opponent):
        next_board, _ = after(board, reserves, opponent, move)
        if check_winner(next_board, opponent) == opponent:
            return True
    return False


def evaluate(board, reserves, player, opponent):
    """Estimate a non-terminal position from ``player``'s perspective.

    Only visible top pieces count in a winning line.  A line containing both
    players is blocked, whereas a line containing pieces from just one player
    remains useful.  The values intentionally make three-in-a-row threats much
    more important than diffuse board control.
    """
    owners, sizes = top_view(board)
    score = 0
    line_value = (0, 2, 12, 80, 0)

    for line in LINES[WIN_LENGTH]:
        mine = sum(owners[r, c] == player for r, c in line)
        theirs = sum(owners[r, c] == opponent for r, c in line)
        if theirs == 0:
            score += line_value[mine]
        elif mine == 0:
            score -= line_value[theirs]

    # Larger visible pieces can cover more opposing pieces.  Prefer keeping
    # large pieces in reserve slightly, so they remain available as replies.
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            owner = owners[r, c]
            if owner == player:
                score += int(sizes[r, c]) + 1
            elif owner == opponent:
                score -= int(sizes[r, c]) + 1
    score += int(reserves[player, LARGE]) * 2
    score -= int(reserves[opponent, LARGE]) * 2
    return score

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        moves = get_valid_moves(board, reserves, player)
        survivors = []
        safe = []

        for move in moves:
            next_board, next_reserves = after(board, reserves, player, move)
            winner = check_winner(next_board, player)

            if winner == player:
                return move
            if winner == opponent:
                # Moving a top piece can reveal the opponent's completed line.
                # That loss has priority over any line made by this move.
                continue

            candidate = (move, next_board, next_reserves)
            survivors.append(candidate)
            if not opponent_has_immediate_win(next_board, next_reserves, opponent):
                safe.append(candidate)

        # If every move permits an immediate opponent win, still prefer a move
        # that does not itself lose by the uncovering rule.  This fallback also
        # guarantees a legal move if no alternative exists.
        pool = safe or survivors
        if not pool:
            return moves[0]

        best_move = pool[0][0]
        best_score = evaluate(pool[0][1], pool[0][2], player, opponent)
        for move, next_board, next_reserves in pool[1:]:
            score = evaluate(next_board, next_reserves, player, opponent)
            if score > best_score:
                best_move, best_score = move, score

        return best_move
