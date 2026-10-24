"""The referee.

World plays one game between two agents it never trusts and writes down everything that
happened in a GameRecord. It talks to the agents only through runners (runners.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from constants import (ALLOW_MOVES, WIN_LENGTH, MAX_PLIES, STEP_TIME_LIMIT, STEP_HARD_LIMIT,
                       MAX_STRIKES)
from helpers import (Move, new_game, copy_state, check_endgame, must_pass, check_move_validity,
                     execute_move, random_move)
from runners import AgentLoadError, AgentInitTimeout, StepResult


@dataclass
class GameRecord:
    """Everything the tournament needs to know about one game.

    World fills it in while the game runs; at the end the tournament writes it to a JSON file
    (to_dict) and the standings and team reports are computed from those files alone. The seed
    and the move list are enough to replay the game exactly.
    """
    seed: int                      # the game's seed. new_game(seed) rebuilds the same start position
                                   # (obstacle and removed size), so a game can be replayed from it.
    obstacle: tuple[int, int]      # (row, col) of the blocked cell, chosen by the seed
    removed_size: int              # the size taken away from both players' reserves: SMALL, MEDIUM or LARGE
    rules: dict                    # the rule flags of this game, e.g. {"allow_moves": True, "win_length": 3}
    players: dict                  # who played which seat: {"1": {"agent": "random_agent"}, "2": {...}}.
                                   # The keys are strings because JSON keys are always strings.
    game_id: str = ""              # the tournament's id for this game, e.g. "g000123"; "" elsewhere
    winner: int | None = None      # None while the game runs; then 1 or 2, or 0 for a draw
    reason: str = ""               # why it ended, e.g. "line", "revealed", "no_moves", "exhausted",
                                   # "ply_cap", or a forfeit: "hard_timeout", "crash", "too_many_strikes",
                                   # "init_failed", "init_timeout". "" while the game runs.
    moves: list = field(default_factory=list)                    # one dict per ply, in order: a move
                                   # {"ply", "player", "src", "dest", "size", "agent_time", "wall_time",
                                   # "flags", "detail"} or a pass {"ply", "player", "pass": True}
    strikes: dict = field(default_factory=lambda: {1: 0, 2: 0})  # strikes per player in this game
                                   # (illegal move, late move, exception); one more than MAX_STRIKES forfeits
    init_time: dict = field(default_factory=lambda: {1: 0.0, 2: 0.0})   # unit: seconds. How long each
                                   # agent took to get ready (runner.start)
    max_rss_mb: dict = field(default_factory=lambda: {1: None, 2: None})  # unit: MB. Peak memory of each
                                   # agent's process; None with InProcessRunner (no separate process)
    log_paths: dict = field(default_factory=lambda: {1: None, 2: None})   # the file holding each agent's
                                   # printed output; None with InProcessRunner (prints go to the terminal)
    load_error: str | None = None  # the error text when a runner failed to start (an error or a
                                   # timeout), e.g. "P2 team_042/student_agent: ..."; else None
    forfeit_detail: str | None = None   # the error text of the ply that forfeited the game: a crash's
                                   # last 2 kB of output, or the traceback of a 4th-strike exception; else None
    engine_version: str = "dev"    # which version of the engine played the game; the tournament puts
                                   # the git hash here
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))  # when the record
                                   # was created, e.g. "2026-12-01T10:22:05"
    duration_s: float = 0.0        # unit: seconds. How long the whole game took; set by World.finish

    @property
    def plies(self) -> int:
        """Return the number of plies played so far. A pass counts as a ply."""
        return len(self.moves)

    def add(self, ply: int, player: int, move: Move, agent_time: float, wall_time: float,
            flags: list[str], detail: str | None = None) -> None:
        """Add one played move to the end of `moves`.

        Takes: ply, player           - the ply number (0 is the first ply) and who moved, 1 or 2.
               move                  - the Move that was played: the agent's own move, or the
                                       random move that replaced it after a strike.
               agent_time, wall_time - seconds, copied from the agent's StepResult.
               flags                 - what went wrong in this ply, e.g. ["soft_timeout"]; [] if nothing.
               detail                - extra text for the team report, e.g. the end of a traceback.
        """
        self.moves.append({"ply": ply, "player": player, "src": None if move.src is None else list(move.src),
                           "dest": list(move.dest), "size": move.size, "agent_time": round(agent_time, 6),
                           "wall_time": round(wall_time, 6), "flags": list(flags), "detail": detail})

    def add_pass(self, ply: int, player: int) -> None:
        """Add a pass to the end of `moves`: `player` could not place a piece and skipped ply `ply`."""
        self.moves.append({"ply": ply, "player": player, "pass": True})

    def finish(self, winner: int, reason: str) -> None:
        """Store the result: winner is 1 or 2, or 0 for a draw; reason is e.g. "line"."""
        self.winner = winner
        self.reason = reason

    def time_stats(self, player: int) -> dict:
        """Return the step times of one player, in seconds.

        Takes:   player - 1 or 2.
        Returns: {"max": ..., "mean": ..., "init": ...}. Passes are left out; max and mean
                 are 0.0 if the player never moved.
        """
        times = [m["agent_time"] for m in self.moves if m["player"] == player and "agent_time" in m]
        return {"max": max(times) if times else 0.0,
                "mean": (sum(times) / len(times)) if times else 0.0,
                "init": self.init_time[player]}

    def to_dict(self) -> dict:
        """Return the record as plain dicts, lists, strings and numbers, ready for json.dumps.

        Values kept per player are keyed by the strings "1" and "2", because JSON keys are
        always strings.
        """
        return {"schema": 1, "game_id": self.game_id, "seed": self.seed, "obstacle": list(self.obstacle),
                "removed_size": self.removed_size, "rules": dict(self.rules), "players": self.players,
                "winner": self.winner, "reason": self.reason, "plies": self.plies, "moves": self.moves,
                "strikes": {str(p): self.strikes[p] for p in (1, 2)},
                "time": {str(p): self.time_stats(p) for p in (1, 2)},
                "max_rss_mb": {str(p): self.max_rss_mb[p] for p in (1, 2)},
                "log_paths": {str(p): self.log_paths[p] for p in (1, 2)},
                "load_error": self.load_error, "forfeit_detail": self.forfeit_detail,
                "engine_version": self.engine_version, "started_at": self.started_at,
                "duration_s": round(self.duration_s, 3)}

    def summary_line(self) -> str:
        """Return the result of the game in one line.

        Example: "seed 7 obstacle (3, 3) removed 1 | P1 greedy_agent wins by line after 9 plies | strikes 0/0"
        """
        names = {p: self.players[str(p)]["agent"] for p in (1, 2)}
        result = "draw" if self.winner == 0 else f"P{self.winner} {names[self.winner]} wins"
        return (f"seed {self.seed} obstacle {self.obstacle} removed {self.removed_size} | "
                f"{result} by {self.reason} after {self.plies} plies | strikes {self.strikes[1]}/{self.strikes[2]}")


class World:
    """The referee of one game. Usage: record = World(seed, runner_1, runner_2).play()

    World owns the only real board. Each ply it checks whether the game is over, hands the
    agent to move a copy of the position, judges what comes back (legal? on time? did it
    crash?) and applies the move itself. An agent can never touch the real board or the
    record; the worst it can do is lose the game.
    """

    def __init__(self, seed: int, runner_1, runner_2, *, allow_moves: bool = ALLOW_MOVES,
                 win_length: int = WIN_LENGTH, display: bool = False, display_delay: float = 0.4,
                 max_plies: int = MAX_PLIES, soft_limit: float = STEP_TIME_LIMIT,
                 hard_limit: float = STEP_HARD_LIMIT, max_strikes: int = MAX_STRIKES,
                 record_moves: bool = True):
        """Build the start position and start both agents.

        seed         : chooses the obstacle and the removed size (new_game), and seeds the referee's
                       own random generator for substitute moves, so a replay is identical.
        runner_1/2   : the runners (runners.py) of player 1, who moves first, and player 2.
        Everything after the `*` must be passed by name:
        allow_moves, win_length : the rule flags of this game (win_length 3 or 4, else ValueError).
                       World passes them to every helper call and never reads constants again.
                       Defaults: the values in constants.py when world.py was imported; the
                       simulator and the tournament always pass them explicitly.
        display      : draw the board in a matplotlib window (in-process runners only).
        display_delay: seconds to pause after each drawn ply.
        max_plies    : the game is a draw ("ply_cap") after this many plies; it can only lower
                       constants.MAX_PLIES.
        soft_limit   : seconds. A step() slower than this is a strike and its move is thrown away.
                       It does not apply to an agent that waits for a person (the human agent).
        hard_limit   : seconds. No answer by then: the agent is killed and forfeits
                       (only SubprocessRunner can enforce it).
        max_strikes  : strikes forgiven per game; the next one forfeits.
        record_moves : False leaves the per-move entries out of the record (passes are still kept).
        If an agent fails to start, the game is already over when __init__ returns
        (reason "init_failed" or "init_timeout").
        """
        if win_length not in (3, 4):
            raise ValueError(f"win_length must be 3 or 4, got {win_length!r}")
        self.rules = {"allow_moves": bool(allow_moves), "win_length": int(win_length)}
        self.seed = int(seed)
        self.board, self.reserves, self.obstacle, self.removed = new_game(self.seed)
        # Used only to pick a random move after a strike. `seed ^ 0xA5A5` flips some bits of the
        # seed, so this generator differs from the one that chose the obstacle.
        self.rng = np.random.default_rng(self.seed ^ 0xA5A5)
        self.runner_1, self.runner_2 = runner_1, runner_2
        self.max_plies, self.soft_limit, self.hard_limit, self.max_strikes = max_plies, soft_limit, hard_limit, max_strikes
        self.record_moves = record_moves
        self.player, self.ply, self.over = 1, 0, False
        self.strikes = {1: 0, 2: 0}
        self.record = GameRecord(self.seed, self.obstacle, self.removed, dict(self.rules),
                                 {"1": {"agent": runner_1.name}, "2": {"agent": runner_2.name}})
        self._t_start = time.perf_counter()
        self.display, self.display_delay, self.ui = display, display_delay, None

        for seat, runner in ((1, runner_1), (2, runner_2)):
            failure = None
            try:
                runner.start(seat, dict(self.rules))
            except AgentLoadError as exc:            # AgentInitTimeout is a subclass, so it lands here too
                failure = exc
            self.record.init_time[seat] = runner.stats().get("init_time", 0.0) or 0.0
            if failure is not None:
                self.record.load_error = f"P{seat} {runner.name}: {str(failure)[-2000:]}"
                reason = "init_timeout" if isinstance(failure, AgentInitTimeout) else "init_failed"
                self.finish(3 - seat, reason)
                return
        if display:
            from ui import UIEngine                  # imported here, so matplotlib loads only when needed
            self.ui = UIEngine(self)
            self.ui.render(self.board, self.reserves, self.record)

    def runner(self, player: int):
        """Return the runner of `player` (1 or 2)."""
        return self.runner_1 if player == 1 else self.runner_2

    def finish(self, winner: int, reason: str) -> None:
        """End the game: store the winner (1 or 2, or 0 for a draw) and the reason, e.g. "line".

        If a window is open, the final board is drawn with the result.
        """
        self.over = True
        self.record.finish(winner, reason)
        self.record.duration_s = time.perf_counter() - self._t_start
        if self.ui is not None:                  # draw the final position with the result
            self.ui.render(self.board, self.reserves, self.record)

    def _normalize_move(self, returned):
        """Check the type of what an agent returned, before its legality is checked.

        Takes:   returned - whatever the agent's step() returned.
        Returns: (move, note). A Move gives (that Move, None). A tuple (src, dest, size) that makes
                 a valid Move gives (that Move, "tuple_move"). Anything else gives (None, None).
        Example: (None, (0, 0), 2) -> (Move(src=None, dest=(0, 0), size=2), "tuple_move")
        """
        if isinstance(returned, Move):
            return returned, None
        if isinstance(returned, (tuple, list)) and len(returned) == 3:
            try:
                return Move(returned[0], returned[1], returned[2]), "tuple_move"
            except (ValueError, TypeError):
                return None, None
        return None, None

    def _strike(self, player: int, flag: str, flags: list[str]) -> bool:
        """Give `player` one strike and add `flag` (e.g. "illegal_move") to `flags`.

        Returns True if the player now has more than max_strikes strikes and so forfeits.
        """
        self.strikes[player] += 1
        self.record.strikes[player] = self.strikes[player]
        flags.append(flag)
        return self.strikes[player] > self.max_strikes

    def adjudicate(self, player: int, result: StepResult):
        """Decide what happens with one answer of an agent.

        Takes:   player - the player who was asked, 1 or 2.
                 result - the StepResult from the player's runner.
        Returns: (move, flags, forfeit_reason).
                 move           - the Move to play: the agent's own move, or a random legal move
                                  after a strike; None if the player forfeits.
                 flags          - what went wrong, e.g. ["illegal_move"]; [] if nothing.
                 forfeit_reason - "" if the game goes on, else e.g. "crash" or "too_many_strikes".
        """
        flags: list[str] = []
        if result.kind == "hard_timeout":
            return None, ["hard_timeout"], "hard_timeout"
        if result.kind == "crash":
            return None, ["crash"], "crash"
        am = self.rules["allow_moves"]
        too_many = False                         # True once a strike is one more than max_strikes
        if result.kind == "error":
            too_many = self._strike(player, "exception", flags)
        else:
            move, note = self._normalize_move(result.move)
            if note:
                flags.append(note)
            # A late move is thrown away even when it is legal, so the time check comes first.
            # A person at the keyboard (the human agent) may take as long as they need.
            slow = result.agent_time > self.soft_limit and not getattr(self.runner(player), "interactive", False)
            if slow:
                too_many = self._strike(player, "soft_timeout", flags)
            elif move is None or not check_move_validity(self.board, self.reserves, player, move, allow_moves=am):
                # A board move that would be legal if board moves were on gets its own flag, so
                # the team report can explain it.
                if move is not None and not am and move.src is not None \
                        and check_move_validity(self.board, self.reserves, player, move, allow_moves=True):
                    too_many = self._strike(player, "move_not_allowed", flags)
                else:
                    too_many = self._strike(player, "illegal_move", flags)
            else:
                return move, flags, ""
        if too_many:
            return None, flags, "too_many_strikes"
        # The referee's own generator, so a replay with the same seed picks the same move.
        return random_move(self.board, self.reserves, player, self.rng, allow_moves=am), flags, ""

    def step(self) -> None:
        """Play exactly one ply. Does nothing once the game is over.

        In order: check whether the game is over; record a pass if the player cannot place a
        piece; otherwise ask the agent for a move, judge the answer, and apply the move.
        """
        if self.over:
            return
        p, opp = self.player, 3 - self.player
        am, wl = self.rules["allow_moves"], self.rules["win_length"]
        over, winner, reason = check_endgame(self.board, self.reserves, p, self.ply, allow_moves=am, win_length=wl)
        # A lower cap than constants.MAX_PLIES, in check_endgame's order: after a line, before no-moves.
        if self.ply >= self.max_plies and reason not in ("line", "revealed"):
            over, winner, reason = True, 0, "ply_cap"
        if over:
            self.finish(winner, reason); return
        if must_pass(self.board, self.reserves, p, allow_moves=am):
            self.record.add_pass(self.ply, p)        # the agent is not asked
            self.ply += 1; self.player = opp
            if self.ui is not None:
                self.ui.render(self.board, self.reserves, self.record)
            return

        snapshot = copy_state(self.board, self.reserves)    # the agent gets copies, never the real board
        result = self.runner(p).step(snapshot, p, opp, self.hard_limit)
        move, flags, forfeit = self.adjudicate(p, result)
        if move is None:
            if result.error:                         # e.g. a crash's log tail, for the team report
                self.record.forfeit_detail = f"P{p} {self.runner(p).name}: {result.error[-2000:]}"
            self.finish(opp, forfeit); return
        execute_move(self.board, self.reserves, p, move, allow_moves=am)
        if self.record_moves:
            detail = result.error[-2000:] if result.kind == "error" and result.error else None
            if "illegal_move" in flags or "move_not_allowed" in flags:
                detail = repr(result.move)[:200]          # what the agent returned, for the team report
            self.record.add(self.ply, p, move, result.agent_time, result.wall_time, flags, detail)
        self.ply += 1; self.player = opp
        if self.ui is not None:
            self.ui.render(self.board, self.reserves, self.record)

    def play(self) -> GameRecord:
        """Play the whole game and return the finished GameRecord.

        Both runners are stopped at the end, even if something went wrong during the game.
        """
        try:
            while not self.over:
                self.step()
        finally:
            for seat, runner in ((1, self.runner_1), (2, self.runner_2)):
                try:                                 # one broken runner must not keep the other running
                    runner.stop()
                    st = runner.stats()
                    self.record.max_rss_mb[seat] = st.get("max_rss_mb")
                    self.record.log_paths[seat] = st.get("log_path")
                except Exception:
                    pass
        return self.record
