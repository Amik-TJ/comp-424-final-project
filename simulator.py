"""Play games from the command line.

Examples:
  python simulator.py --player_1 student_agent --player_2 random_agent             # one game, every ply logged
  python simulator.py --player_1 human_agent --player_2 greedy_agent --display      # play it yourself
  python simulator.py --player_1 student_agent --player_2 random_agent --autoplay   # 100 games and a summary
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

import constants
from helpers import new_game
from runners import InProcessRunner
from world import World

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SIZE_FROM_LETTER = {"S": 0, "M": 1, "L": 2}    # --removed S/M/L -> SMALL/MEDIUM/LARGE


def get_args(argv=None):
    """Read the command-line options.

    Takes:   argv - a list of option strings, e.g. ["--seed", "5"]; None reads the real command line.
    Returns: an object with one attribute per option, e.g. args.seed, args.player_1.
    A bad value (an odd --autoplay_runs, an --obstacle off the board) stops the program with a message.
    """
    parser = argparse.ArgumentParser(description="Goblet Gobblers simulator")
    parser.add_argument("--player_1", type=str, default="random_agent")
    parser.add_argument("--player_2", type=str, default="random_agent")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--obstacle", type=str, default=None, help="r,c  (overrides the seed's obstacle)")
    parser.add_argument("--removed", type=str, default=None, choices=["S", "M", "L"])
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display_delay", type=float, default=0.4)
    parser.add_argument("--display_save", action="store_true", default=False)
    parser.add_argument("--display_save_path", type=str, default="plots/")
    parser.add_argument("--autoplay", action="store_true", default=False)
    parser.add_argument("--autoplay_runs", type=int, default=100)
    parser.add_argument("--allow_moves", action=argparse.BooleanOptionalAction, default=constants.ALLOW_MOVES)
    parser.add_argument("--win_length", type=int, choices=[3, 4], default=constants.WIN_LENGTH)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--soft_limit", type=float, default=constants.STEP_TIME_LIMIT)
    parser.add_argument("--hard_limit", type=float, default=constants.STEP_HARD_LIMIT)
    parser.add_argument("--max_plies", type=int, default=constants.MAX_PLIES)
    parser.add_argument("--json", type=str, default=None, help="append one GameRecord per game to this .jsonl")
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.autoplay and args.autoplay_runs % 2 != 0:
        parser.error("--autoplay_runs must be even (games are played in seat-swapped pairs)")
    if args.obstacle is not None:
        try:
            r, c = (int(x) for x in args.obstacle.split(","))   # ValueError: not two whole numbers
        except ValueError:
            parser.error(f"--obstacle must be two numbers 'r,c', got {args.obstacle!r}")
        if not (0 <= r < constants.BOARD_SIZE and 0 <= c < constants.BOARD_SIZE):
            parser.error(f"--obstacle {args.obstacle} is off the board (rows and columns are 0 to {constants.BOARD_SIZE - 1})")
    return args


def find_seed(start: int, obstacle=None, removed=None) -> int:
    """Return the first seed, counting up from `start`, whose game has the requested setup.

    Takes:   start    - the first seed to try.
             obstacle - the (row, col) of the obstacle, or None for any cell.
             removed  - the removed size, SMALL, MEDIUM or LARGE, or None for any size.
    Returns: the seed. Raises ValueError if no seed matches within 10 000 tries.
    Example: find_seed(0, obstacle=(1, 2), removed=SMALL) -> 24
    """
    for seed in range(start, start + 10_000):        # 48 setups exist: a match comes long before
        _, _, obs, rem = new_game(seed)
        if (obstacle is None or tuple(obs) == tuple(obstacle)) and (removed is None or rem == removed):
            return seed
    raise ValueError(f"no seed from {start} on gives obstacle={obstacle} removed={removed}")


def make_runner(spec: str, strict: bool):
    """Return the runner for one player.

    Takes: spec   - a registry name, e.g. "random_agent", or a path to a .py file.
           strict - True runs the agent in its own process with the tournament limits
                    (SubprocessRunner); False runs it inside this program (InProcessRunner).
    """
    if strict:
        from runners import SubprocessRunner
        return SubprocessRunner(spec)
    return InProcessRunner(spec)


class Simulator:
    """Plays games with the options from get_args(): one logged game (run), or many games
    followed by a summary (autoplay)."""

    def __init__(self, args):
        """Store the options and choose the base seed.

        Takes: args - the result of get_args().
        The base seed is --seed, or a random number. With --obstacle or --removed, it moves up
        to the first seed whose game has that setup. With --json, the file is opened for appending.
        """
        self.args = args
        self.rules = {"allow_moves": args.allow_moves, "win_length": args.win_length}
        base = args.seed if args.seed is not None else int(np.random.default_rng().integers(1_000_000))
        obstacle = tuple(int(x) for x in args.obstacle.split(",")) if args.obstacle else None
        removed = SIZE_FROM_LETTER[args.removed] if args.removed else None
        self.base_seed = find_seed(base, obstacle, removed) if (obstacle or removed is not None) else base
        self.json_out = open(args.json, "a") if args.json else None

    def play_one(self, seed: int, p1: str, p2: str, display: bool = False):
        """Play one game and return its GameRecord.

        Takes: seed    - the game's seed.
               p1, p2  - the agents of player 1 and player 2: registry names or .py paths.
               display - True draws the board in a window.
        With --json, the record is also added to that file as one line. With --autoplay, an agent
        that needs a person at the keyboard (human_agent) stops the program with an error.
        """
        a = self.args
        world = World(seed, make_runner(p1, a.strict), make_runner(p2, a.strict), **self.rules,
                      display=display, display_delay=a.display_delay, max_plies=a.max_plies,
                      soft_limit=a.soft_limit, hard_limit=a.hard_limit)
        if a.autoplay and any(getattr(r, "interactive", False) for r in (world.runner_1, world.runner_2)):
            raise SystemExit("error: human_agent needs a person at the keyboard; it cannot be used with --autoplay")
        record = world.play()
        if self.json_out:
            self.json_out.write(json.dumps(record.to_dict()) + "\n"); self.json_out.flush()
        return record

    def run(self):
        """Play one game, log every ply and the result, and return the GameRecord.

        With --quiet only the result is logged. With --display the window stays open at the
        end until you close it.
        """
        a = self.args
        logger.info("Rules: allow_moves=%s win_length=%d max_plies=%d", a.allow_moves, a.win_length, a.max_plies)
        record = self.play_one(self.base_seed, a.player_1, a.player_2, display=a.display)
        if record.load_error:
            logger.error("failed to load: %s", record.load_error[-300:])
        for m in ([] if a.quiet else record.moves):   # --quiet: only the result line
            if m.get("pass"):
                logger.info("ply %2d P%d passes", m["ply"], m["player"])
            else:
                logger.info("ply %2d P%d %s -> %s size %d  (%.3f s) %s", m["ply"], m["player"],
                            m["src"], m["dest"], m["size"], m["agent_time"], " ".join(m["flags"]))
        logger.info(record.summary_line())
        if a.display:                                # keep the final board on screen until the window is closed
            import matplotlib.pyplot as plt
            plt.ioff(); plt.show()
        return record

    @staticmethod
    def _tally(tally, record, label_1, label_2):
        """Add the result of one game to the totals.

        Takes: tally            - {label: totals}, as built by autoplay().
               record           - the finished GameRecord.
               label_1, label_2 - the tally labels of the agents in seat 1 and seat 2.
        """
        for seat, label in ((1, label_1), (2, label_2)):
            t = tally[label]
            if record.winner == 0:
                t["D"] += 1
            elif record.winner == seat:
                t["W"] += 1
                if seat == 1:
                    t["as_p1_wins"] += 1
            else:
                t["L"] += 1
                if record.reason in ("hard_timeout", "crash", "too_many_strikes", "init_failed", "init_timeout"):
                    t["forfeits"] += 1
            for m in record.moves:
                if m.get("player") == seat and "agent_time" in m:   # passes have no agent_time
                    t["times"].append(m["agent_time"])
                    for f in m["flags"]:
                        t["strikes"][f] = t["strikes"].get(f, 0) + 1

    @staticmethod
    def print_summary(tally, pairs):
        """Print the totals, two lines per agent: its results, then its step times and strikes.

        Takes: tally - {label: totals}, as built by autoplay().
               pairs - the number of seat-swapped pairs played (each pair is two games).
        """
        games = 2 * pairs
        width = max(len(s) for s in tally)
        for spec, t in tally.items():
            pct = 100.0 * (t["W"] + 0.5 * t["D"]) / games
            print(f"{spec:<{width}} : W {t['W']:3d}  D {t['D']:3d}  L {t['L']:3d}  -> {pct:5.1f} %   "
                  f"as P1: {t['as_p1_wins']}/{pairs}   as P2: {t['W'] - t['as_p1_wins']}/{pairs}")
        for spec, t in tally.items():
            times = t["times"] or [0.0]
            s = t["strikes"]
            print(f"step time  {spec:<{width}}: max {max(times):.2f} s  mean {sum(times)/len(times):.2f} s  "
                  f"soft-timeouts {s.get('soft_timeout', 0)}  illegal {s.get('illegal_move', 0) + s.get('move_not_allowed', 0)}  "
                  f"exceptions {s.get('exception', 0)}  forfeits {t['forfeits']}")

    def autoplay(self):
        """Play --autoplay_runs games in seat-swapped pairs, print a summary, and return the totals.

        Pair i uses seed base_seed + i twice: once with player_1 moving first, once with
        player_2 moving first, so both agents get the same boards.
        """
        a = self.args
        pairs = a.autoplay_runs // 2
        specs = (a.player_1, a.player_2)
        # One set of totals per command-line player, even when both players are the same agent.
        labels = specs if specs[0] != specs[1] else (f"{specs[0]} (player_1)", f"{specs[1]} (player_2)")
        tally = {label: {"W": 0, "D": 0, "L": 0, "as_p1_wins": 0, "times": [], "strikes": {}, "forfeits": 0}
                 for label in labels}
        print(f"Rules: allow_moves={a.allow_moves}  win_length={a.win_length}  max_plies={a.max_plies}")
        print(f"Pairs: {pairs} ({a.autoplay_runs} games)   configs: seeds {self.base_seed}..{self.base_seed + pairs - 1}")
        logging.disable(logging.CRITICAL)              # no per-ply log lines during autoplay
        try:
            for i in range(pairs):
                for first, second in ((0, 1), (1, 0)):          # one seed, seats swapped
                    record = self.play_one(self.base_seed + i, specs[first], specs[second])
                    self._tally(tally, record, labels[first], labels[second])
        finally:
            logging.disable(logging.NOTSET)
        self.print_summary(tally, pairs)
        return tally


def main(argv=None):
    """Run the simulator: one game, or an autoplay series with --autoplay.

    Takes: argv - a list of option strings; None reads the real command line.
    """
    args = get_args(argv)
    rules = {"allow_moves": args.allow_moves, "win_length": args.win_length}
    constants.configure_rules(**rules)          # before any agent is imported, since agents read the flags
    import agents  # noqa: F401                  # registers the built-in agents
    if not args.strict:
        logger.warning("in-process mode: hard time limits are NOT enforced; use --strict for the tournament harness")
    if args.autoplay and args.display:
        logger.warning("--display is ignored in autoplay mode")
    if args.display and args.display_save:       # save every frame of the window as an image
        import ui
        ui.SAVE_DIR = args.display_save_path
    sim = Simulator(args)
    if args.autoplay:
        sim.autoplay()
    else:
        sim.run()


if __name__ == "__main__":
    main()
