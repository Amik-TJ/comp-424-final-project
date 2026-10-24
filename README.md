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
to create and activate your virtual environment. On Windows, the two commands are `py -3.11 -m venv env_name` and
`env_name\Scripts\activate`. Note: You will need to activate your virtual environment every time you start a new shell.
It should appear as:
```
(env_name) <username@machine>:~/your/path/to/project$
```

Ensure that you have properly installed pip and python in your venv. To check this, you can run (on Windows, `where pip`):
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
The tournament runs Python 3.11 with numpy 1.26 (the version `requirements.txt` installs). Use Python 3.11 too: the
validator warns when it runs on another version.

## What's in this repository

You edit only `agents/student_agent.py`. The tournament runs your file with its own copy of every other file,
so changes elsewhere have no effect on your grade.

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
This is a summary. [`game_rules.txt`](game_rules.txt) has the same rules in numbered points.

In the boards below, `1M` is player 1's medium piece, and `2S<1M` is a stack listed from bottom to top: player 2's
small under player 1's medium. `.` is an empty cell and `##` the obstacle. The simulator prints boards the same way.

### The board and the pieces
- The board has 4 × 4 cells. A cell is written `(row, col)`; `(0, 0)` is the top-left cell.
- One cell is an **obstacle**. No piece can ever go there.
- There are three sizes of pieces: small (S) < medium (M) < large (L).
- Each player starts from 3 small, 3 medium and 3 large pieces. One piece of a randomly chosen size is then removed
  from **both** players, so each player has 8 pieces and both players have the same pieces.
- The board starts empty, except for the obstacle. The pieces you have not placed yet are your **reserve**.
- The obstacle cell and the removed size come from the game's seed: the same seed always gives the same start.
  There are 16 × 3 = 48 possible starts.
- Nothing is hidden. Your agent sees every stack, covered pieces included, and both reserves.

### A turn
Player 1 moves first; then the players take turns. The agents swap seats from one game to the next (see Scoring),
so each agent moves first equally often. A turn (a **ply**) is exactly one of:
- **Place** a piece from your reserve on a cell that is empty or whose top piece is **strictly smaller**. You may
  cover ("gobble") any smaller piece, your own or your opponent's, at any time.
- **Move** your own top piece from the board to another cell, by the same rule (this exists because
  `ALLOW_MOVES = True` this term). Lifting a piece shows the piece under it again. You may move a piece even while you
  still have pieces in your reserve. A piece on the board never goes back to the reserve.

Three cases that come up often:
- A covered piece cannot move, even if it is yours, until the piece above it leaves. On a cell `1S<2M`, player 1
  cannot move the `1S`; player 1 can cover the whole cell with a large piece.
- A piece must go to a different cell. Lifting it and putting it back on the same cell is illegal.
- A piece covers only strictly smaller pieces: M on M, S on M and anything on L are illegal. A large piece can never be
  covered.

### Winning
A **line** is `WIN_LENGTH` cells in a row, a column or a diagonal whose top pieces all belong to one player. Only top
pieces count: a covered piece counts for nobody. A line through the obstacle can never be completed. With 4 in a row
there are 10 lines (4 rows, 4 columns, 2 diagonals); with 3 in a row there are 24.

**Win check, after every move.** Let P be the player who just moved and O the opponent. Look at the top piece of
every cell. If O has WIN_LENGTH in a row, O wins — this is how "you uncovered my line and did not cover it again"
is scored, and it also decides the case where both players have a line. Otherwise, if P has WIN_LENGTH in a row,
P wins. Otherwise play goes on.

**Example: gobbling and revealing.** Player 1 is to move.
```
0    2S<1M      2L       .       .
1        .      ##       .       .
2        .       .       .       .
3        .       .       .       .
           0       1       2       3
```
Player 1 moves its medium from (0, 0) to (0, 2). Lifting it shows the `2S` again, so the top pieces of row 0 are now
2, 2, 1, empty: nobody has a line, and play goes on.

**Example: uncovering a line loses.** Player 1 is to move.
```
0    2S<1L      2M      2L      2M
1        .       .       .       .
2       1M      1S       .      1M
3        .       .       .      ##
           0       1       2       3
```
If player 1 lifts the `1L` from (0, 0), the `2S` under it shows, and row 0 belongs to player 2 unless the large piece
lands back in that row.

| Player 1 moves the `1L` from (0, 0) to | Result |
|---|---|
| (2, 2) | Player 1 completes row 2, but row 0 now reads 2, 2, 2, 2. O is checked first: **player 2 wins**. |
| (0, 1) or (0, 3), covering a `2M` | Row 0 reads 2, 1, 2, 2 or 2, 2, 2, 1: no line, play goes on. |
| (0, 2) | Illegal: nothing can cover a large piece. |

### How a game ends
- A player has a line: that player wins (the win check above).
- The player to move has no legal move: that player **loses**. This is rare.
- 100 plies (50 for each player) are played and nobody has won: a **draw**. A line made on the 100th ply still wins.
- There is no repetition rule. Repeating the same position does not end the game; only the 100-ply limit does.

### Scoring
- A win gives 1 point, a draw 0.5, a loss 0. A forfeit is a loss.
- Games are played in **pairs**. Both games of a pair start from the same configuration (same obstacle, same removed
  size), and the agents swap seats, so each agent moves first once. A match between two agents has 2, 4 or 6 games;
  the course staff will announce the number. `--autoplay` plays the same way.

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

Example, the first board of the rules above:
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
        # Does the opponent have a reply that wins? any() stops at the first one it finds.
        opponent_wins = any(check_winner(after(b, r, opponent, om)[0], opponent) == opponent
                            for om in get_valid_moves(b, r, opponent))
        if not opponent_wins:
            safe.append(mv)
    pool = safe or moves                        # `safe` if it is not empty, else every move
    return pool[int(self.rng.integers(len(pool)))]
```

A plain alpha-beta search in Python (no move ordering, no transposition table) reaches depth 2 to 3 in the 2-second
budget on a recent laptop; move ordering and a transposition table (`helpers.state_key`) are where the gains are.

## Submission contract
Your submission is **one file**, `student_agent.py`, plus `authors.yaml`. It must satisfy all of:

| # | Requirement |
|---|---|
| C-01 | File name exactly `student_agent.py`, UTF-8 encoded, no BOM, ≤ 2 MB. |
| C-02 | Valid Python for the grading interpreter: **Python 3.11**. No `match` on 3.9, no PEP 695 syntax, etc. — run the validator with the same interpreter version. |
| C-03 | Imports only from the Python standard library, `numpy`, `helpers`, `constants`, `store`, `agents.agent`; nothing else: no `scipy`, `torch`, `numba`, `pandas`, no `world`, `simulator`, `runners`, `ui`, no files of your own. |
| C-04 | Exactly one class decorated `@register_agent("student_agent")` — the string must be exactly that. The class name may be anything (`StudentAgent` recommended). |
| C-05 | That class subclasses `agents.agent.Agent`; its `__init__` can be called without arguments and calls `super().__init__()`. |
| C-06 | It implements `step(self, board, reserves, player, opponent)` (any other parameter needs a default), returning a `helpers.Move` (not `None`, not a tuple, not a coroutine). |
| C-07 | `__init__` returns within 30 s; every `step` returns within 2.0 s wall-clock (measure with `time.perf_counter()`; leave a safety margin). |
| C-08 | Peak memory under 500 MB. |
| C-09 | Keep module-level code to imports, constants, and function and class definitions; any other work at import time counts toward the 30 s start-up limit, and a loop there that never ends is a FAIL. |
| C-10 | No file I/O, no network, no extra threads or processes (the validator warns about file I/O); no `sys.exit()` / `quit()` / `os._exit()`, no `exec`, `eval`, `compile` or `input`; `sys.setrecursionlimit` at most 10 000; no monkey-patching of `helpers`, `numpy` or `constants`. |
| C-11 | The agent must work in either seat (`player == 1` or `2`) and must only return legal moves. Illegal moves, exceptions and timeouts each cost a strike; four strikes forfeit the game. |
| C-12 | `python validate_submission.py agents/student_agent.py` must print `PASS` with the official interpreter. The tournament runs the same checks; a `FAIL` means the agent does not enter the tournament and receives the "poor" performance grade. |
| C-13 | The agent must respect the rule set of the game: read `ALLOW_MOVES` and `WIN_LENGTH` from `constants` (never assign them, never call `configure_rules`), return a board move only when `ALLOW_MOVES` is `True`, and evaluate lines of `WIN_LENGTH`. The tournament uses the values announced in the handout. An agent that also works with the other values is welcome but not required. |

**Imports.** `allowed_imports.py` holds the rule, and the validator and the tournament both use it. Your file may
import:
- any module of the Python standard library (`math`, `time`, `random`, `collections`, `heapq`, `itertools`, …), as
  listed by the Python that runs it;
- `numpy` and any part of it (`numpy.linalg`, …);
- from this project, only `helpers`, `constants`, `store` and `agents.agent`.

Nothing else: not the other files of this project (`world`, `runners`, `simulator`, `ui`, `validate_submission`,
`allowed_imports`), not the other agents (`agents.random_agent`, `agents.greedy_agent`, …), no package the tournament
machine does not have (`torch`, `scipy`, `matplotlib`, …), and no files of your own. A module you may import is not
always one you may use for everything: C-10 still forbids file I/O, network, and extra threads or processes.

### Check before you submit
```bash
python validate_submission.py agents/student_agent.py
```
It must print `RESULT: PASS`. It checks the file without running it (imports, decorator, signatures, forbidden calls),
then plays four sandboxed games against the random agent and reports your step times and strikes. Passing the
validator does not mean your agent is good; it means the tournament can run it. On Windows, if the sandboxed games
cannot start, add `--in_process`; the static checks are the same everywhere.

### How to submit
Submit exactly two files to the myCourses group assignment: `student_agent.py` and `authors.yaml`. Do not rename
them and do not zip them. Before submitting, `python validate_submission.py agents/student_agent.py` must print `RESULT: PASS`.

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

Without `--strict`, the simulator counts late moves as strikes but enforces neither the 10 s limit nor the memory
limit. `--strict`, the validator and the tournament enforce all of them. On macOS and Windows, `--strict` prints
`WARNING: not available on this platform: …` once for each protection the system lacks, and the games still run;
the tournament machine (Linux) enforces every limit.

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

Feel free to contact us in an Ed thread.
