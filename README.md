# Goblet Gobblers — COMP 424 final project

**Project description & template:** https://www.overleaf.com/project/6adc1036c0b369a348637ee4/share#f7b97c81b2af34dc012345849ea4608f6f9df6465bc80718

## Setup
To set up your Python environment, we highly suggest using a virtual environment to keep your dependencies in order. Run:
```bash
python3.11 -m venv env_name
```
and then:
```bash
source env_name/bin/activate
```
to create and activate your virtual environment. Note: You will need to activate your virtual environment every time
you start a new shell. It should appear as:
```
(env_name) <username@machine>:~/your/path/to/project$
```

Ensure that you have properly installed pip and python in your venv. To check this, you can run:
```bash
which pip
```
If you get something like:
```
/usr/bin/pip
```
rather than:
```
/path/to/your/venv/bin/pip
```
then you may need to run:
```bash
python -m pip install --upgrade pip
```

To set up the game, clone this repository and install the dependencies. Be sure to have your virtual environment activated.
```bash
pip install -r requirements.txt
```

## What's in this repository

You edit only `agents/student_agent.py`. We grade your agent with our own copy of every other file, not yours.
If you change any other file and your agent depends on that change, it will fail on our machine, and you will lose
all the points for your agent.

| File | What it does |
|---|---|
| `agents/student_agent.py` | **Your agent**: the only file you edit and submit. Write your search in `step()`. |
| `agents/agent.py` | The base class `Agent`. The docstring of `step()` explains its arguments, what to return, and the time limit. |
| `helpers.py` | The game rules as functions for your agent: legal moves (`get_valid_moves`), playing a move (`execute_move`), the win check (`check_winner`, `check_endgame`), copying a position (`copy_state`), and the `Move` type. |
| `constants.py` | Board size, piece sizes, time and memory limits, and the rule flags of the game (`ALLOW_MOVES`, `WIN_LENGTH`). Read them; never change them. |
| `world.py` | The referee: sets up the board from a seed, asks each agent for a move, checks it (legal? on time?), plays it, and records the game. |
| `runners.py` | The bridge between the referee and an agent: runs the agent inside the same program, or, with `--strict`, in its own process under the tournament's time and memory limits. |
| `simulator.py` | The command line: play one game, watch it, play it yourself, or run many games with `--autoplay`. |
| `ui.py` | Draws the board in a window (`--display`). |
| `validate_submission.py` | Checks your `student_agent.py` before you submit. It must print `RESULT: PASS`. |
| `allowed_imports.py` | The modules your agent may import. The validator and the tournament both use it. |
| `store.py` | The agent registry behind `@register_agent`. |
| `game_rules.txt` | The rules of the game in numbered points, for the rule set used this term. |
| `authors.yaml` | Your team's names and McGill IDs. Submit it together with `student_agent.py`. |
| `requirements.txt` | The Python packages to install. |

### Agents you can play against

| Agent | How it plays |
|---|---|
| `random_agent` | A random legal move. The first opponent to beat. |
| `greedy_agent` | Looks one move ahead: wins at once if it can, otherwise avoids moves that let the opponent win next. `agents/greedy_agent.py` is also a short example of how to use the helpers. |
| `human_agent` | You, at the keyboard: `--player_1 human_agent --display`. Type `p S 1 2` to place a small piece on row 1, column 2, `m 0 0 1 2` to move your top piece, `q` to quit. |
| `student_agent` | Your agent. The template plays at random until you write your own search. |

## Playing a game
The simulator needs two agents. Several are provided; your job is to write your own in `agents/student_agent.py`.

```bash
python simulator.py --player_1 random_agent --player_2 greedy_agent            # one game, text log
python simulator.py --player_1 human_agent  --player_2 greedy_agent --display  # play yourself
python simulator.py --player_1 student_agent --player_2 random_agent --autoplay --autoplay_runs 100
python simulator.py --player_1 student_agent --player_2 greedy_agent --autoplay --strict   # the tournament harness
python simulator.py --player_1 student_agent --player_2 greedy_agent --seed 42            # a fixed configuration
```

`--autoplay` plays games in **pairs**: the same obstacle and reserve with the seats swapped, so the first-move
advantage cancels out. `--strict` runs each agent in its own process with the tournament's time and memory limits;
use it before you submit. `--display` shows the board (stacks are drawn as rings: a small ring inside a big one is a
piece that has been gobbled).

The board is generated from a seed: one random obstacle cell and one random size removed from each player's
3 small / 3 medium / 3 large. `--seed N` reproduces a configuration; `--obstacle r,c --removed S|M|L` picks one.
`python simulator.py -h` lists every option.

## Rules
The rules of the game are in [`game_rules.txt`](game_rules.txt).

**Rule flags in force this term:** `ALLOW_MOVES = True`, `WIN_LENGTH = 4` (read them from `constants`, never
hard-code them). The simulator and the validator can also play other values (`--no-allow_moves`, `--win_length 3`)
if you want to experiment; the tournament uses only the values above.

## Writing your agent

### The position: `board` and `reserves`
```
board[r, c, s]   r, c ∈ {0..3}   s ∈ {0: SMALL, 1: MEDIUM, 2: LARGE}
value ∈ { 0 = EMPTY, 1 = PLAYER_1, 2 = PLAYER_2, 3 = OBSTACLE }
```
- `board` is a numpy array of shape `(4, 4, 3)` and type `int8`. The third axis is the **size slot**, not a stack
  index. A cell holds at most one piece of each size, so `board[r, c, :]` is the whole stack, from small (bottom) to
  large (top).
- The top piece of a cell is the largest `s` with `board[r, c, s] != 0`.
- The obstacle cell has `board[r, c, :] == 3` in all three slots.

Example. The simulator prints a board like this. `1M` is player 1's medium piece, and `2S<1M` is a stack listed from
bottom to top: player 2's small under player 1's medium. `.` is an empty cell and `##` the obstacle.
```
0    2S<1M      2L       .       .
1        .      ##       .       .
2        .       .       .       .
3        .       .       .       .
           0       1       2       3
```
In `board`, three of these cells are:
```
board[0, 0] = [2, 1, 0]    # 2S underneath, 1M on top, no large
board[0, 1] = [0, 0, 2]    # only a 2L
board[1, 1] = [3, 3, 3]    # obstacle
```

`reserves` is a numpy array of shape `(3, 3)` and type `int8`:
```
reserves[player, s] = number of pieces of size s that `player` has not placed yet
reserves[0, :] is unused and always 0  (row index == player number)
```
At the start, `reserves[1]` and `reserves[2]` are both `[3, 3, 3]` minus one piece of the removed size.

### A move: `helpers.Move`
```python
@dataclass(frozen=True, slots=True)
class Move:
    src: tuple[int, int] | None    # None: place a piece from your reserve; a cell: move your top piece from there
    dest: tuple[int, int]          # the cell the piece goes to
    size: int                      # SMALL, MEDIUM or LARGE (0, 1, 2); for a move, the size of the piece you lift
```
`Move(None, (1, 2), LARGE)` places a large piece on row 1, column 2. `Move((0, 0), (1, 2), MEDIUM)` moves your medium
from (0, 0) to (1, 2). A `Move` cannot be changed after it is made, and it can be a dictionary key. Making one with a
cell off the board or a size other than 0, 1 or 2 raises `ValueError`.

### The helpers
`helpers.py` holds the rules as functions. Only `execute_move` changes the arrays you give it.

| Function | What it gives |
|---|---|
| `get_top(board, r, c)` | `(owner, size)` of the top piece of cell `(r, c)`; `(EMPTY, -1)` for an empty cell, `(OBSTACLE, -1)` for the obstacle. |
| `top_view(board)` | `(owners, sizes)`, two 4 × 4 arrays for the top piece of every cell: the owner (0 empty, 1, 2, 3 obstacle) and the size (-1 where empty or obstacle). |
| `get_valid_moves(board, reserves, player)` | Every legal move of `player`, as a list of `Move`: placements first (small, medium, large), then board moves. Always in the same order. |
| `check_move_validity(board, reserves, player, move)` | `True` if `move` is legal now. Never raises: anything that is not a `Move` gives `False`. |
| `execute_move(board, reserves, player, move)` | Plays the move by changing `board` and `reserves` in place. Raises `ValueError` for an illegal move, and then changes nothing. |
| `check_winner(board, last_mover)` | The winner right after `last_mover` moved, by the win check: the opponent if it has a line, else `last_mover` if it has one, else 0. |
| `has_line(owners, player)` | `True` if `player` owns every cell of a line in `owners`, the first array of `top_view`. |
| `check_endgame(board, reserves, player_to_move, ply)` | `(is_over, winner, reason)` just before `player_to_move` plays, when `ply` plies have been played. `winner` is 0 for a draw or a game that goes on; `reason` is `"line"`, `"revealed"` (the mover uncovered the opponent's line), `"ply_cap"`, `"no_moves"`, `"exhausted"` (only when `ALLOW_MOVES` is `False`), or `""` while the game goes on. |
| `must_pass(board, reserves, player)` | `True` if `player` must skip the turn. Only possible when `ALLOW_MOVES` is `False`. |
| `random_move(board, reserves, player, rng=None)` | A random legal move, or `None` if there is none. `rng` is an optional `np.random.default_rng(...)` generator. |
| `copy_state(board, reserves)` | New copies of both arrays, to try a move without changing the real ones. |
| `state_key(board, reserves, player)` | A `bytes` value for the whole position (board, reserves, player to move): a dictionary key, e.g. for a transposition table. |
| `new_game(seed)` | `(board, reserves, obstacle, removed_size)`: the start of the game with that seed, as the simulator makes it. |
| `render_text(board, reserves)` | The board and both reserves as text, as the simulator prints them. |
| `LINES[3]`, `LINES[4]` | Every line of 3 cells (24) and of 4 cells (10), each a tuple of `(row, col)` cells. `LINES[WIN_LENGTH]` gives the lines of this game. Read it; never change it. |

The functions that depend on a rule flag also take `allow_moves=` or `win_length=` keyword arguments. Leave them out:
the default, `None`, reads the flags of the current game from `constants`.

### What the referee guarantees
1. `step` is called only when you have at least one legal move. With `ALLOW_MOVES = True` you are asked on every one of
   your turns, so your k-th call (counting from 0) is ply `2 * k + player - 1`. With `ALLOW_MOVES = False`, a player
   who cannot place a piece is skipped without being asked.
2. `board` and `reserves` are fresh copies at every call. You may change them and keep them.
3. One agent object plays one whole game, so you can keep data between calls (e.g. a transposition table). A new
   object is made for every game, so nothing carries over from one game to the next.
4. `player` stays the same for the whole game.
5. In the tournament (and with `--strict`), `step` runs in the main thread of its own process. `__init__` has already
   returned, and nothing else runs in that process.
6. `constants.ALLOW_MOVES` and `constants.WIN_LENGTH` already hold the rules of the game when your file is imported,
   and they do not change during the game.

### The template
`agents/student_agent.py` as you receive it:
```python
"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

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
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        return random_move(board, reserves, player)
```

### An example: the greedy agent
`agents/greedy_agent.py` looks one move ahead with the helpers. `after(board, reserves, player, move)` is a small
function of that file: it plays `move` on copies made with `copy_state` and returns them.
```python
def step(self, board, reserves, player, opponent):
    moves = get_valid_moves(board, reserves, player)
    safe = []                                   # moves after which the opponent cannot win at once
    for mv in moves:
        b, r = after(board, reserves, player, mv)
        w = check_winner(b, player)
        if w == player:
            return mv                           # this move wins the game now
        if w == opponent:
            continue                            # lifting this piece uncovers the opponent's line
        # Keep the move only if no reply of the opponent wins at once.
        opponent_wins = any(check_winner(after(b, r, opponent, om)[0], opponent) == opponent
                            for om in get_valid_moves(b, r, opponent))
        if not opponent_wins:
            safe.append(mv)
    pool = safe or moves                        # every move loses at once: play any of them
    return pool[int(self.rng.integers(len(pool)))]
```

A plain alpha-beta search in Python (no move ordering, no transposition table) reaches depth 2 to 3 in the 2-second
budget on a recent laptop; move ordering and a transposition table (`helpers.state_key`) are where the gains are.

## Submission Guidelines and Rules
Each team submits two files: `student_agent.py` and `authors.yaml`. The file `student_agent.py` must comply with
every rule below. The validator refers to these rules by number in its messages.

### Rules

| Rule | Requirement |
|---|---|
| C-01 | The file must be named exactly `student_agent.py`, must be encoded in UTF-8 without a byte-order mark, and must not exceed 2 MB. |
| C-02 | The file must be valid Python 3.11, the version used for grading. Syntax introduced in later versions, such as the type parameter syntax of Python 3.12 (PEP 695), is not permitted. The validator must be run with Python 3.11. |
| C-03 | The file may import only the Python standard library, `numpy`, and the project modules `helpers`, `constants`, `store` and `agents.agent`. All other imports are prohibited, including other packages (for example `scipy`, `torch`, `numba` and `pandas`), the other project files (`world`, `simulator`, `runners` and `ui`), and files of your own. |
| C-04 | The file must contain exactly one class decorated with `@register_agent("student_agent")`, with exactly this string. The class may have any name; `StudentAgent` is recommended. |
| C-05 | This class must be a subclass of `agents.agent.Agent`. Its `__init__` method must be callable without arguments and must call `super().__init__()`. |
| C-06 | The class must implement `step(self, board, reserves, player, opponent)`. Any additional parameter must have a default value. The method must return a `helpers.Move`; it must not return `None`, a tuple or a coroutine. |
| C-07 | `__init__` must return within 30 seconds, and every call to `step` must return within 2.0 seconds of wall-clock time. Time must be measured with `time.perf_counter()`, with a safety margin. |
| C-08 | The peak memory use of the agent must remain below 500 MB. |
| C-09 | Module-level code must be limited to imports, constants, and function and class definitions. Any other code executed at import time counts toward the 30-second initialization limit. A module-level loop that never ends causes the validator to report `FAIL`. |
| C-10 | The agent must not access files or the network, and must not start additional threads or processes. The validator reports file access as a warning. The agent must not call `sys.exit()`, `quit()`, `os._exit()`, `exec`, `eval`, `compile` or `input`. A call to `sys.setrecursionlimit` must not set a limit above 10 000. The agent must not modify (monkey-patch) `helpers`, `numpy`, `constants` or `store`. |
| C-11 | The agent must play in either seat (`player` equal to 1 or 2) and must return only legal moves. Each illegal move, exception or late move counts as one strike. A fourth strike in a game forfeits that game. |
| C-12 | The command `python validate_submission.py agents/student_agent.py`, run with Python 3.11, must print `RESULT: PASS`. The tournament applies the same checks. An agent that receives `FAIL` does not enter the tournament and receives the "poor" performance grade. |
| C-13 | The agent must follow the rule set of the game. It must read `ALLOW_MOVES` and `WIN_LENGTH` from `constants`, must never assign them, and must never call `configure_rules`. It may return a board move only when `ALLOW_MOVES` is `True`, and it must evaluate lines of length `WIN_LENGTH`. The tournament uses the values announced in the handout. Support for other values is welcome but not required. |

### Permitted imports
The import rule is defined in `allowed_imports.py`; the validator and the tournament both apply it. The file
`student_agent.py` may import:
- any module of the Python standard library (for example `math`, `time`, `random`, `collections`, `heapq` and
  `itertools`), as listed by the interpreter that runs the file;
- `numpy` and any of its submodules (for example `numpy.linalg`);
- from this project, only `helpers`, `constants`, `store` and `agents.agent`.

All other imports are prohibited: the other files of this project (`world`, `runners`, `simulator`, `ui`,
`validate_submission` and `allowed_imports`), the other agents (for example `agents.random_agent` and
`agents.greedy_agent`), packages that are not installed on the tournament machine (for example `torch`, `scipy` and
`matplotlib`), and files of your own. A permitted import does not permit every use of the module: rule C-10 still
prohibits file access, network access, and additional threads or processes.

### Validation before submission
Before submitting, run:
```bash
python validate_submission.py agents/student_agent.py
```
The command must print `RESULT: PASS`. The validator first examines the file without running it (imports, decorator,
signatures and prohibited calls). It then plays four sandboxed games against `random_agent` and reports the step
times and strikes. A `PASS` confirms only that the tournament can run the agent; it says nothing about the agent's
strength. If the sandboxed games cannot start on your system, add the option `--in_process`; the static checks are
the same in both modes.

### Submission procedure
Each team submits exactly two files to the myCourses group assignment: `student_agent.py` and `authors.yaml`. The
files must not be renamed and must not be compressed (for example, into a ZIP archive). The validator must print
`RESULT: PASS` for `student_agent.py` before the files are submitted.

## Time and memory

### Limits
| Limit | Value |
|---|---|
| Agent construction (`__init__`) wall time | 30 s, once per game |
| `step()` wall time, soft limit | 2.0 s per call, measured inside the agent process around the call |
| `step()` wall time, hard limit | 10 s; exceeding it kills the agent process and forfeits the game |
| Strikes tolerated per game | 3; each strike replaces the move by a random legal move; the 4th strike forfeits the game |
| Memory (RSS) per agent process | 500 MB (the sandbox's cap is set 200 MB higher, to leave room for Python and numpy) |
| stdout/stderr per agent per game | 1 MB, truncated afterwards |
| Illegal move returned | counts as a strike; replaced by a random legal move |
| Exception raised in `step()` | counts as a strike; replaced by a random legal move |
| Crash of the agent process (segfault, `os._exit`, killed by a limit) | forfeit of that game |

A late `step()` counts as a strike too. "Forfeit" means a loss for that agent and a win for the opponent.

On macOS and Windows, `--strict` prints `WARNING: not available on this platform: …` once for each protection the
system lacks, and the games still run; the tournament machine (Linux) enforces every limit.

### Maximum memory measurements
On Linux, measure using:
```bash
/usr/bin/time -v python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
```
and then you should see, among other lines:
```
        Maximum resident set size (kbytes): 36100
```
where the maximum resident set size is the most RAM that you have used (in kilobytes). Without `--strict` both agents
run inside the simulator, so this number is the simulator and both agents together: an upper bound for your agent.
On macOS, use `/usr/bin/time -l` instead; it prints the "maximum resident set size" in bytes. The validator also
reports the peak memory of your agent's own process, on its `sandbox load` line (`rss … MB`).

### The grading command
Before you submit, make sure your agent runs with the exact command we use for grading:
```bash
python simulator.py --player_1 random_agent --player_2 student_agent --autoplay --strict
```
It plays 100 games (50 pairs) against `random_agent` under the tournament's limits, each agent in its own process,
and prints a summary: your score, your step times and your strikes. With the template it takes less than a minute.
An agent that thinks for most of its 2 seconds at every move needs about that long per move, so plan for 100 games
× your moves per game × your time per move.

## FAQ

**My agent is slow as player 2.** Both seats are played: in every pair of games your agent is player 1 once and
player 2 once, with the same 2-second limit. Time both seats. `--autoplay` does, and its summary shows your slowest
step over all games.

**Can I use my own helper file?** No. You submit one file (C-03): put every function and class you need inside
`student_agent.py`. An import of a file of your own fails the validator.

**Can I change `constants.py`?** No (C-13). The tournament runs your file with its own copy of every other file, so a
change has no effect there, and assigning `ALLOW_MOVES` or `WIN_LENGTH` in your file fails the validator. Read the
values; never write them.

**Why did I get a strike?** Your `step()` did one of these:
- took more than 2 s (shown as `soft_timeout`);
- returned something that is not a legal `Move` (`illegal_move`);
- returned a board move while `ALLOW_MOVES` is `False` (`move_not_allowed`);
- raised an exception (`exception`).

The referee then plays a random legal move for you. The simulator's log prints the flag next to the ply, and the
validator shows the first case of each kind (for an exception, with the line of your file). A 4th strike in one game, a `step()` that
takes more than 10 s, or a crash of your agent's process forfeits the game.

**How is the tournament scored?** A win gives 1 point, a draw 0.5 and a loss 0; a forfeit is a loss. Every match is
played in pairs of games: both games of a pair start from the same configuration, with the seats swapped. A match has
2, 4 or 6 games; the course staff will announce the number. The project description explains how the results count
toward your grade.

**Can I play two versions of my agent against each other?** Yes. Copy your current agent to a file outside `agents/`
and give its path instead of an agent name:
```bash
cp agents/student_agent.py my_old_agent.py
python simulator.py --player_1 student_agent --player_2 my_old_agent.py --autoplay
```

## Issues? Bugs? Questions?

First, open an issue on the GitHub page of this repository. If that does not solve the problem, post in an Ed
thread, or contact fahad.amik@mail.mcgill.ca on Microsoft Teams.
