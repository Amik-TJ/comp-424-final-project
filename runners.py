"""Agent runners.

The referee (world.py) never calls an agent's step() itself. It asks a runner, and the runner
answers with a StepResult. Two runners exist, both with the methods listed in AgentRunner:
  InProcessRunner  - the agent runs inside the referee's own process
  SubprocessRunner - the agent runs in its own child process, with limits
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import sys
import time
import traceback
import uuid
from typing import NamedTuple, Protocol

from constants import INIT_TIME_LIMIT, current_rules
from store import AGENT_REGISTRY


class StepResult(NamedTuple):
    """What a runner hands back to the referee after asking an agent for one move.

    The runner only reports what happened; it never checks whether the move is legal. The
    referee does that in World.adjudicate: play the move, give a strike, or forfeit.
    Example: StepResult("ok", Move(None, (1, 2), 0), 0.84, 0.91, None)
    Keep the five fields in this order: SubprocessRunner.step builds it by position.
    """
    kind: str            # what happened, one of four words:
                         #   "ok"           step() returned a value; the referee still checks it
                         #                  (too late -> strike, illegal -> strike, else it is played)
                         #   "error"        step() raised an exception -> strike, a random move is played
                         #   "hard_timeout" step() ran past the hard limit and was killed -> forfeit
                         #   "crash"        the agent's process died (os._exit, memory, segfault) -> forfeit
    move: object         # what step() returned, NOT checked yet. Usually a helpers.Move, but it can be
                         # None, a tuple or junk. Always None unless kind == "ok". SubprocessRunner
                         # rebuilds it in the referee's process: a Move (also for a subclass of Move),
                         # a tuple, None, or, for anything else, a stand-in whose repr() is the agent's.
    agent_time: float    # unit: seconds. Time spent inside agent.step() itself, measured in the agent's
                         # own process. The soft limit STEP_TIME_LIMIT (2 s) is judged on this number.
    wall_time: float     # unit: seconds. Total time the referee waited for the answer, including sending
                         # the board to the agent's process and the reply back. Recorded, never judged.
                         # With InProcessRunner it equals agent_time.
    error: str | None    # "error": the error text (usually a traceback). "crash": the last 2 kB of the
                         # agent's log. None for "ok" and "hard_timeout".


class AgentLoadError(Exception):
    """The agent could not be made ready to play. Raised by a runner's start().

    Causes: the file does not exist; importing it fails (syntax error, sys.exit() at import, or,
    in the sandbox only, a forbidden import); no agent class can be found in it; its __init__
    raises; or, in the sandbox only, the agent's process reports other rule flags than the
    referee asked for.
    World catches it: that agent loses before ply 0 with reason "init_failed",
    and the message is kept in GameRecord.load_error for the team report.
    """


class AgentInitTimeout(AgentLoadError):
    """The agent took longer than INIT_TIME_LIMIT (30 s) to get ready, mostly in __init__.

    World scores it as reason "init_timeout". Because it is a kind of AgentLoadError,
    `except AgentLoadError` catches it too; World then tells the two apart with
    isinstance(exc, AgentInitTimeout).
    """


class AgentRunner(Protocol):
    """The methods the referee (World) uses to talk to one agent.

    A Protocol is a checklist, not a parent class: World accepts any object that has `name`
    and these four methods, without inheriting from AgentRunner. InProcessRunner and
    SubprocessRunner both fit it.
    The `...` after each def means "no body here, only the signature".
    One game, in order: start() once, step() once per ply, stop() once; stats() at any time.
    A runner may also have `interactive`: True when its agent waits for a person at the keyboard
    (the human agent). World never gives such an agent a strike for being slow. A runner
    without this attribute counts as not interactive.
    """
    name: str            # the agent's name in records and logs, e.g. "random_agent", "team_042/student_agent"

    # Load the agent and build it (run its __init__), once per game, before ply 0.
    # seat: 1 or 2. rules: this game's flags, {"allow_moves": ..., "win_length": ...}.
    # Raises AgentLoadError, or AgentInitTimeout when getting ready takes too long.
    def start(self, seat: int, rules: dict) -> None: ...

    # Ask the agent for one move. snapshot: (board, reserves), fresh copies the agent may change.
    # player: the agent's number (1 or 2); opponent: the other number. hard_limit: seconds before
    # the agent is killed (only SubprocessRunner can enforce it). A misbehaving agent never makes
    # this raise: the failure comes back inside the StepResult. (Ctrl-C and the human agent's
    # `q` are the exceptions: they still end the program.)
    def step(self, snapshot, player, opponent, hard_limit) -> StepResult: ...

    # The game is over for this agent: kill its process if it has one. World calls it in a
    # `finally` block, so it also runs after an error, and calling it twice must be harmless.
    def stop(self) -> None: ...

    # {"init_time": seconds, "max_rss_mb": peak memory in MB or None, "log_path": the file holding
    # the agent's printed output, or None}. World copies these into the GameRecord.
    def stats(self) -> dict: ...


def _is_file_spec(agent_spec: str) -> bool:
    """True if agent_spec is a file path, False if it is a registry name.

    It is a path if it ends with ".py" or contains "/" or "\\" (Windows).
    Examples: "random_agent" -> False, "team_042/student_agent.py" -> True.
    """
    # Only the text is checked, never the disk: a folder named "random_agent" must not
    # turn the registry name into a path.
    return agent_spec.endswith(".py") or "/" in agent_spec or "\\" in agent_spec


def _agent_subclasses_in(module) -> list[type]:
    """List the Agent subclasses defined in `module`, each once, in file order.

    Classes the module only imports (e.g. RandomAgent) are left out.
    Used by load_agent_class when a file has no usable @register_agent.
    """
    from agents.agent import Agent   # imported here, not at the top: see load_agent_class
    found = []
    for value in vars(module).values():
        # isinstance() first: issubclass() raises TypeError if value is not a class.
        # __module__ names the file a class was defined in, so imported classes, and Agent
        # itself, fail that test. `not in found` skips an alias, e.g. `StudentAgent = MyAgent`.
        if isinstance(value, type) and issubclass(value, Agent) \
                and value.__module__ == module.__name__ and value not in found:
            found.append(value)
    return found


def load_agent_class(agent_spec: str, registry_name: str = "student_agent") -> type:
    """Return the agent class that agent_spec names. Does not create the agent.

    agent_spec   : a registry name, e.g. "random_agent", or a path to a .py file,
                   e.g. "submissions/team_042/student_agent.py".
    registry_name: for a file, the name it should register with @register_agent.
    For a file, the class returned is the first one found of:
      1. the class it registered as registry_name;
      2. the only class it registered, under another name (prints a warning);
      3. the only Agent subclass it defines (prints a warning).
    AGENT_REGISTRY is left as it was, even when loading fails.
    Raises AgentLoadError: unknown name, missing file, file fails to import, or no class found.
    Not caught: a broken file inside agents/ (e.g. the student's agents/student_agent.py).
    `import agents` below raises its own error, so the student sees the traceback, as last year.
    Example: load_agent_class("random_agent") -> RandomAgent
    """
    # Imported here, not at the top: importing an agent runs its code, which reads the rule
    # flags, so it must happen after configure_rules(). It also registers the built-in agents.
    import agents  # noqa: F401
    if not _is_file_spec(agent_spec):
        if agent_spec in AGENT_REGISTRY:
            return AGENT_REGISTRY[agent_spec]
        raise AgentLoadError(f"agent '{agent_spec}' is not registered; registered: {sorted(AGENT_REGISTRY)}")

    path = os.path.abspath(agent_spec)
    if not os.path.isfile(path):
        raise AgentLoadError(f"agent file not found: {path}")
    # A new module name for every file, so several files can be loaded in one process.
    module_name = f"submission_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:                     # the file name has no suffix Python can import
        raise AgentLoadError(f"not a Python source file: {path}")
    module = importlib.util.module_from_spec(spec)

    # Run the file with an empty registry. Every submission registers "student_agent", like the
    # template in agents/, and register_agent refuses a name that is already taken. What the
    # file registered is copied out, then the registry is put back as it was.
    saved = dict(AGENT_REGISTRY)
    AGENT_REGISTRY.clear()
    sys.modules[module_name] = module    # some code, e.g. @dataclass, looks the module up here while it runs
    try:
        spec.loader.exec_module(module)  # runs the file, top to bottom
        registered = dict(AGENT_REGISTRY)
    except BaseException as exc:         # not just Exception: sys.exit() at import is a load error too
        sys.modules.pop(module_name, None)
        if isinstance(exc, KeyboardInterrupt):   # Ctrl-C must still stop the program
            raise
        raise AgentLoadError(f"import of {path} failed:\n{traceback.format_exc()}") from exc
    finally:
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(saved)

    if registry_name in registered:
        return registered[registry_name]
    if len(registered) == 1:
        name, cls = next(iter(registered.items()))
        print(f"WARNING: {path} registered '{name}', expected '{registry_name}'; using it anyway")
        return cls
    classes = _agent_subclasses_in(module)
    if len(classes) == 1:
        print(f"WARNING: {path} has no usable @register_agent; using class {classes[0].__name__}")
        return classes[0]
    sys.modules.pop(module_name, None)
    raise AgentLoadError(f"{path}: no class registered as '{registry_name}' (registered: {sorted(registered)}, "
                         f"Agent subclasses: {[c.__name__ for c in classes]})")


def agent_display_name(agent_spec: str) -> str:
    """Return the short name shown for an agent in records and logs.

    A registry name stays as it is. A path becomes "<folder>/<file name without .py>".
    Examples: "random_agent" -> "random_agent",
              "submissions/team_042/student_agent.py" -> "team_042/student_agent".
    """
    if not _is_file_spec(agent_spec):
        return agent_spec
    path = os.path.abspath(agent_spec)
    folder = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{folder}/{stem}"


class InProcessRunner:
    """Runs the agent inside the referee's own process: step() is a plain function call.

    Fast and simple. Used for development, --display and the human agent. It cannot enforce
    the hard time limit or the memory limit: an endless loop in step() freezes the whole program.
    agent_spec: a registry name ("random_agent") or a path to a .py file ("agents/student_agent.py").
    registry_name: the name a .py file is expected to register with @register_agent.
    """

    def __init__(self, agent_spec: str, registry_name: str = "student_agent"):
        self.agent_spec = agent_spec
        self.registry_name = registry_name
        self.name = agent_display_name(agent_spec)
        self.agent = None        # the agent object. Created in start(), not here, so that a crash
                                 # or a slow __init__ is timed and costs that game, not the program.
        self.init_time = 0.0     # unit: seconds. How long start() took to get the agent ready.
        self.interactive = False # True when the agent waits for a person (autoplay is False);
                                 # set by start(). World then never strikes it for being slow.

    def start(self, seat: int, rules: dict) -> None:
        """Load the agent's class and create the agent. Called once per game, before the first move.

        seat : unused here; kept so that both runners have the same start().
        rules: this game's rule flags. Must equal constants.current_rules(), because an
               in-process agent reads the flags from constants.
        Sets self.agent, self.interactive (True for an agent with autoplay False, e.g. the human
        agent), and self.init_time (seconds spent loading the file and in __init__; set even
        when start fails, because World records it either way).
        Raises AgentLoadError   : loading the file or __init__ failed.
               AgentInitTimeout : getting ready took longer than INIT_TIME_LIMIT.
               ValueError       : rules differ from constants (a bug in the caller).
        """
        # ValueError, not AgentLoadError: the caller made the mistake, so the program stops
        # instead of the agent losing the game.
        if rules != current_rules():
            raise ValueError(f"the game's rules {rules} differ from constants {current_rules()}; "
                             "call constants.configure_rules() before starting in-process agents")
        # The clock starts before loading, as in the sandbox, so slow code at the top level of
        # the agent's file counts toward the limit too.
        t0 = time.perf_counter()
        try:
            cls = load_agent_class(self.agent_spec, self.registry_name)
            try:
                self.agent = cls()
            except KeyboardInterrupt:
                raise
            except BaseException as exc:     # includes SystemExit: an agent must not end the program
                raise AgentLoadError(f"{self.name}: __init__ raised:\n{traceback.format_exc()}") from exc
        finally:
            # Also when loading or __init__ fails: the time spent is part of what went wrong.
            self.init_time = time.perf_counter() - t0
        # getattr with a default: an agent that skipped super().__init__() has no autoplay.
        self.interactive = getattr(self.agent, "autoplay", True) is False
        # A slow __init__ cannot be interrupted in-process; it can only be refused once it returns.
        if self.init_time > INIT_TIME_LIMIT:
            raise AgentInitTimeout(f"{self.name}: getting ready took {self.init_time:.1f} s > {INIT_TIME_LIMIT} s")

    def step(self, snapshot, player, opponent, hard_limit) -> StepResult:
        """Ask the agent for one move and report what happened. Does not check the move.

        snapshot  : (board, reserves), copies the agent may change freely.
        player    : the agent's player number, 1 or 2. opponent: the other number.
        hard_limit: ignored; a running function call cannot be stopped in-process.
        Returns StepResult "ok" with whatever step() returned, or "error" with the traceback
        if step() raised, sys.exit() included. agent_time and wall_time are equal here.
        Only two exceptions reach the caller: KeyboardInterrupt (Ctrl-C), and SystemExit
        from an agent whose autoplay is False (the human player typed q).
        Raises RuntimeError if called before start() or after stop().
        """
        if self.agent is None:           # the caller's bug, not the agent's: no strike
            raise RuntimeError(f"{self.name}: step() called before start() or after stop()")
        board, reserves = snapshot
        t0 = time.perf_counter()
        try:
            move = self.agent.step(board, reserves, player, opponent)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            dt = time.perf_counter() - t0
            # getattr with a default: an agent that skipped super().__init__() has no autoplay.
            if isinstance(exc, SystemExit) and getattr(self.agent, "autoplay", True) is False:
                raise
            return StepResult("error", None, dt, dt, traceback.format_exc())
        dt = time.perf_counter() - t0
        return StepResult("ok", move, dt, dt, None)

    def stop(self) -> None:
        """End the game for this agent: drop it, so no later game can reuse it. Safe to call twice."""
        self.agent = None

    def stats(self) -> dict:
        """Return {"init_time": seconds, "max_rss_mb": None, "log_path": None}.

        Memory and log are None: the agent has no process of its own, and its prints go to the terminal.
        """
        return {"init_time": self.init_time, "max_rss_mb": None, "log_path": None}


# --- the agent host (child side)

_MAX_TEXT = 20_000             # unit: characters. The most of a traceback that one reply carries;
                               # the game record keeps the last 2 000 of them.
_MAX_REPLY_BYTES = 1_000_000   # unit: bytes. The longest reply the referee accepts from an agent process.


def _requested_by_student(frame) -> bool:
    """True if the import being resolved was asked for by the student's own code.

    frame: the frame that called the import hook.
    Frames of Python's import system (importlib) are skipped: the first other frame is the code
    that asked for the import. It is student code when its module name starts with "submission_",
    the prefix load_agent_class gives every submission. A library that imports its own parts,
    such as numpy, or statistics importing decimal, is therefore never refused, even while the
    student's step() is running.
    """
    while frame is not None:
        name = frame.f_globals.get("__name__")
        name = name if isinstance(name, str) else ""
        if not (name == "importlib" or name.startswith(("importlib.", "_frozen_importlib"))):
            return name.startswith("submission_")
        frame = frame.f_back
    return False


def _install_import_guard() -> None:
    """Make every import the student's code asks for raise ImportError unless the allow-list has it.

    Python resolves an import in two stages, so two hooks are needed:
      1. builtins.__import__ is wrapped. Every import statement calls it, and it serves modules
         that are already loaded straight from sys.modules; this is the only place that catches
         `import os`, because os is loaded long before the agent is.
      2. A finder is put first in sys.meta_path. Python asks the finders, in order, where a module
         that is not loaded yet lives; this also catches importlib.import_module("socket").
    Both hooks refuse an import only when _requested_by_student() says the student's code asked
    for it, so the imports that the host and the libraries make for themselves always work.
    Call once, in the agent process, after the host has imported everything it trusts.
    """
    import builtins
    import importlib.abc
    from allowed_imports import is_allowed

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # level > 0 is a relative import inside a library package; only absolute names are checked.
        if level == 0 and _requested_by_student(sys._getframe(1)):
            wanted = [name]
            if name == "agents":   # `from agents import random_agent` names the module after `import`
                wanted += [f"agents.{item}" for item in (fromlist or ())]
            for module_name in wanted:
                if not is_allowed(module_name):
                    raise ImportError(f"module {module_name!r} is not allowed in student_agent.py")
        return real_import(name, globals, locals, fromlist, level)

    class GuardFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not is_allowed(fullname) and _requested_by_student(sys._getframe(1)):
                raise ImportError(f"module {fullname!r} is not allowed in student_agent.py")
            return None   # allowed, or not asked for by the student: the normal finders take over

    builtins.__import__ = guarded_import
    sys.meta_path.insert(0, GuardFinder())


def _apply_limits(cfg: dict) -> list[str]:
    """Put the agent process in its own process group and set the operating-system limits.

    Runs first in the agent process, before any code the student wrote.
      setsid       : a new process group, so one kill from the parent reaches every process the
                     agent may have started, not just the host.
      RLIMIT_AS    : memory, memory_limit_mb plus 200 MB of headroom for the interpreter and numpy.
                     An allocation past it raises MemoryError inside step().
      RLIMIT_CPU   : CPU seconds for the whole life of the process, enough for every ply to use the
                     full hard limit. Past it, the operating system ends the process with SIGXCPU,
                     and one second later with SIGKILL if SIGXCPU was ignored.
      RLIMIT_FSIZE : the largest file it may write, far above the log cap, as a disk-space backstop.
      RLIMIT_NOFILE: at most 64 open files.
    The hard limits are lowered too (to the soft value, or one second above it for the CPU), because
    a process may raise its own soft limit up to the hard one, but can never raise a hard limit.
    cfg: the settings dict built by SubprocessRunner.start().
    Returns the names of the protections that could not be applied, [] on Linux. The host sends
    them to the referee, which warns once, and carries on, so a game still runs on macOS or Windows.
    """
    missing = []
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            missing.append("setsid")
    else:
        missing.append("setsid")
    try:
        import resource   # Unix only
    except ImportError:
        return missing + ["rlimits"]
    mem = (cfg["memory_limit_mb"] + 200) * 1024 * 1024
    cpu = int(cfg["max_plies"] * cfg["hard_limit"] + cfg["init_limit"]) + 5
    # (name, soft, hard). The CPU hard limit is one second higher: at the hard limit Linux sends
    # SIGKILL, which tells nobody why, while SIGXCPU at the soft limit names the cause.
    for name, value, top in (("RLIMIT_AS", mem, mem), ("RLIMIT_CPU", cpu, cpu + 1),
                             ("RLIMIT_FSIZE", cfg["log_cap_bytes"] * 64, cfg["log_cap_bytes"] * 64),
                             ("RLIMIT_NOFILE", 64, 64)):
        try:
            limit = getattr(resource, name)
            _, hard = resource.getrlimit(limit)
            if hard != resource.RLIM_INFINITY:   # a hard limit can only be lowered, never raised
                value, top = min(value, hard), min(top, hard)
            resource.setrlimit(limit, (value, top))
        except (ValueError, OSError, AttributeError):
            missing.append(name)
    return missing


class _CappedWriter:
    """A replacement for sys.stdout / sys.stderr that writes at most `cap` bytes, then drops the rest.

    print() keeps working for the agent whatever it writes: once the log is full, or the disk
    refuses more, nothing is raised, the text is simply not stored. write() returns the full
    length, as a real file would, so code that checks the return value is not confused.
    fd : the file descriptor to write to, 1 or 2, already pointed at the log file.
    cap: the most bytes this writer will ever write.
    """
    encoding = "utf-8"   # read by code that checks how the stream encodes text

    def __init__(self, fd: int, cap: int):
        self.fd, self.cap, self.written = fd, cap, 0

    def write(self, text: str) -> int:
        if not isinstance(text, str):   # the same rule as a real text stream
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        room = self.cap - self.written
        if room > 0:
            try:
                # A character is at most 4 bytes, so cutting first keeps a huge text cheap to encode.
                self.written += os.write(self.fd, text[:room].encode("utf-8", "replace")[:room])
            except OSError:   # the disk is full or the file-size limit is reached: stop writing
                self.written = self.cap
        return len(text)

    def flush(self) -> None:
        pass   # os.write is unbuffered, so there is nothing to flush

    def isatty(self) -> bool:
        return False   # a log file, never a terminal


def _redirect_output(log_path: str, cap: int) -> None:
    """Send everything the agent process prints to its log file instead of the referee's terminal.

    Done at two levels. File descriptors 1 and 2 are pointed at the log, so output from any
    code, C libraries included, lands there. Then sys.stdout and sys.stderr are replaced by a
    _CappedWriter each, so Python-level prints stop at `cap` bytes per stream. The log file is
    created, or emptied if it already exists.
    """
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd not in (1, 2):
        os.close(fd)   # 1 and 2 now point at the log; this third descriptor is no longer needed
    sys.stdout = _CappedWriter(1, cap)
    sys.stderr = _CappedWriter(2, cap)


def _max_rss_mb() -> float | None:
    """Return the peak memory this process has used so far, in MB; None where it cannot be read.

    ru_maxrss is in kilobytes on Linux and in bytes on macOS, hence the two conversions.
    """
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(raw / 1024, 1) if sys.platform != "darwin" else round(raw / (1024 * 1024), 1)
    except Exception:
        return None


def _short_repr(value, limit: int = 200) -> str:
    """Return repr(value) cut to `limit` characters, or a plain description if repr() fails.

    repr() runs the agent's own code when the value is one of its objects, so it may raise
    anything; object.__repr__ never runs agent code.
    Example: _short_repr((x for x in "ab")) -> "<generator object <genexpr> at 0x7f...>"
    """
    try:
        return repr(value)[:limit]
    except BaseException:
        return object.__repr__(value)[:limit]


def _traceback_text() -> str:
    """Return the traceback of the exception being handled, at most _MAX_TEXT characters from its end."""
    try:
        return traceback.format_exc()[-_MAX_TEXT:]
    except BaseException:   # the exception's own str() failed while the traceback was being built
        return "an exception was raised, but its traceback cannot be shown"


def _wire_move(value):
    """Turn what step() returned into plain data that the referee can read safely.

    None                 -> None
    a Move (or subclass) -> {"move": [src, dest, size]}
    a 3-tuple or 3-list  -> {"tuple": [src, dest, size]}, when it makes a valid Move
    anything else        -> {"repr": "..."}, its text only; the referee calls it an illegal move
    Raises an exception for a value that cannot be pickled (a lambda, a generator, a coroutine);
    the caller reports that as an "error".
    Example: Move(None, (1, 2), LARGE) -> {"move": [None, [1, 2], 2]}
    """
    from helpers import Move
    if value is None:
        return None
    if isinstance(value, Move) or (isinstance(value, (tuple, list)) and len(value) == 3):
        try:
            move = Move(value.src, value.dest, value.size) if isinstance(value, Move) else Move(*value)
            kind = "move" if isinstance(value, Move) else "tuple"
            return {kind: [None if move.src is None else list(move.src), list(move.dest), move.size]}
        except Exception:
            pass   # not a well-formed move: sent as text below
    pickle.dumps(value)   # a value that cannot be pickled is reported as an "error", not as a move
    return {"repr": _short_repr(value)}


def _send_reply(conn, *reply) -> None:
    """Send one reply to the referee as JSON text.

    The other direction uses pickle, but this one must not: the referee would have to unpickle
    data from a process that runs the student's code, and unpickling can build any object. JSON
    carries only strings, numbers, None, lists and dicts, and the referee checks every reply.
    """
    conn.send_bytes(json.dumps(reply).encode("utf-8"))


def _serve(conn, agent) -> None:
    """Answer the referee's messages with the agent's moves until it says quit. Never returns.

    Message in                                     Reply out
      ("step", board, reserves, player, opponent)    ["ok", move, seconds] or
                                                     ["error", seconds, traceback]
      ("quit",)                                      ["bye", max_rss_mb], then the process exits
    seconds is the time spent inside agent.step(), measured here, next to the agent; move is
    what _wire_move() makes of the returned value.
    Anything step() raises becomes an "error" reply, SystemExit and KeyboardInterrupt included,
    so an agent cannot end the host with sys.exit(). A return value that cannot be pickled
    (a lambda, a generator) is reported as an "error" too.
    """
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            _send_reply(conn, "bye", _max_rss_mb())
            os._exit(0)
        _, board, reserves, player, opponent = msg
        t0 = time.perf_counter()
        try:
            move = agent.step(board, reserves, player, opponent)
        except BaseException:
            _send_reply(conn, "error", time.perf_counter() - t0, _traceback_text())
            continue
        dt = time.perf_counter() - t0
        try:
            reply = ("ok", _wire_move(move), dt)
        except BaseException as exc:
            reply = ("error", dt, f"the value returned by step() cannot be sent to the referee: {_short_repr(exc)}")
        _send_reply(conn, *reply)


def agent_host_main(conn, cfg: dict) -> None:
    """The program that runs inside SubprocessRunner's child process; the agent lives here.

    Started as the target of a spawned multiprocessing.Process: a fresh interpreter with none
    of the referee's state. The order of the steps matters:
      1. limits, log redirection and chdir into the private work folder, before any student code;
      2. set this game's rule flags, before any agent module is imported and reads them;
      3. import the modules the host relies on, the built-in agents included, while imports are free;
      4. install the import guard, then load the agent's class from its file (a registry name is
         loaded from the file that defines it, like a submission) and build the agent;
      5. send ["ready", init_seconds, rules, missing] with the flags read back from constants, so
         the parent can tell when a flag did not arrive, and the protections this platform lacks;
         then answer step messages until quit.
    Never returns. Exit codes: 0 after ("quit",); 1 when loading or __init__ failed, after
    sending ["load_error", traceback]; 70 on a bug in the host itself.
    conn: the child's end of the pipe to the referee.
    cfg : the settings dict that SubprocessRunner.start() builds, with the keys agent_file,
          registry_name, rules, log_path, work_dir, memory_limit_mb, hard_limit, max_plies,
          init_limit, log_cap_bytes.
    """
    try:
        missing = _apply_limits(cfg)
        _redirect_output(cfg["log_path"], cfg["log_cap_bytes"])
        os.chdir(cfg["work_dir"])
        # The rule flags come first: every agent module reads them when it is imported, and
        # importing agents.agent below imports all the built-in agents.
        import constants
        constants.configure_rules(**cfg["rules"])
        # The modules the host itself relies on are imported before the guard goes up.
        import numpy  # noqa: F401
        import helpers  # noqa: F401
        import agents.agent  # noqa: F401  (runs agents/__init__.py, which registers the built-in agents)
        agent_file, registry_name = cfg["agent_file"], cfg["registry_name"]
        if not _is_file_spec(agent_file):
            # A registry name, e.g. "student_agent": its file is loaded again below, the way a
            # submission is, so that the import guard covers it exactly as in the tournament.
            registry_name = agent_file
            agent_file = sys.modules[load_agent_class(registry_name).__module__].__file__
        _install_import_guard()
        cls = load_agent_class(agent_file, registry_name)
        t0 = time.perf_counter()
        agent = cls()
        init_time = time.perf_counter() - t0
        _send_reply(conn, "ready", init_time, constants.current_rules(), missing)
    except BaseException:
        try:
            _send_reply(conn, "load_error", _traceback_text())
        finally:
            os._exit(1)
    try:
        _serve(conn, agent)
    except BaseException:
        os._exit(70)   # a bug in the host itself, not in the agent; the parent reports a crash


# --- the subprocess runner (parent side)

def _tail(path: str | None, n: int = 2000) -> str:
    """Return the last `n` bytes of the file at `path` as text; "" if it cannot be read.

    Used for the error field of a "crash" StepResult and for a start-up failure: the end of
    the agent's log is where the traceback or the MemoryError usually is.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


class _ReturnedValue:
    """Stands in, on the referee's side, for a value that step() returned and that is not a move.

    Only the value's text crosses from the agent process, so repr() gives back what the agent's
    own repr() gave, and the game record shows exactly what the agent returned.
    """
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


class _BadReply(Exception):
    """A reply from the agent process that does not follow the protocol; the process counts as broken."""


def _non_negative(value) -> bool:
    """True if value is a finite number that is 0 or more, such as a time or a memory size.

    True and False do not count, although Python treats them as the numbers 1 and 0.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _decode_move(wire):
    """Rebuild, on the referee's side, the value that _wire_move() made in the agent process.

    Returns None, a helpers.Move, a (src, dest, size) tuple, or a _ReturnedValue.
    Raises _BadReply if wire is not one of the forms that _wire_move() produces.
    Example: {"tuple": [None, [0, 0], 2]} -> (None, (0, 0), 2)
    """
    from helpers import Move
    if wire is None:
        return None
    if isinstance(wire, dict) and len(wire) == 1:
        (kind, value), = wire.items()
        if kind == "repr" and isinstance(value, str):
            return _ReturnedValue(value[:200])
        if kind in ("move", "tuple") and isinstance(value, list) and len(value) == 3:
            src, dest, size = value
            try:
                fields = (None if src is None else tuple(src), tuple(dest), size)
                move = Move(*fields)   # checks types and ranges, so a bad cell is caught here
            except (TypeError, ValueError):
                raise _BadReply(f"invalid move fields {_short_repr(value, 100)}") from None
            return move if kind == "move" else fields
    raise _BadReply(f"invalid move {_short_repr(wire, 100)}")


def _decode_reply(raw: bytes) -> list:
    """Check one reply from the agent process and return it as a list, e.g. ["bye", 61.3].

    raw: one message, the JSON text written by _send_reply().
    The agent process runs untrusted code, so every reply is checked against the shape of its
    kind; anything else (not JSON, an unknown kind, a value of the wrong type) raises _BadReply.
    In an "ok" reply the move is rebuilt by _decode_move().
    Example: b'["ok", {"move": [null, [1, 2], 0]}, 0.01]' -> ["ok", Move(None, (1, 2), 0), 0.01]
    """
    try:
        reply = json.loads(raw)
    except (ValueError, RecursionError):   # not JSON, not UTF-8, or nested too deeply to read
        raise _BadReply("the reply is not readable JSON") from None
    if not isinstance(reply, list) or not reply:
        raise _BadReply(f"the reply is not a list: {_short_repr(reply, 100)}")
    kind, args = reply[0], reply[1:]
    if kind == "ok" and len(args) == 2 and _non_negative(args[1]):
        return ["ok", _decode_move(args[0]), args[1]]
    if kind == "error" and len(args) == 2 and _non_negative(args[0]) and isinstance(args[1], str):
        return reply
    if kind == "ready" and len(args) == 3 and _non_negative(args[0]) and isinstance(args[1], dict) \
            and all(type(flag) in (bool, int) for flag in args[1].values()) \
            and isinstance(args[2], list) and all(isinstance(m, str) for m in args[2]):
        return reply   # the rules dict is flat, so printing it in an error message is always safe
    if kind == "load_error" and len(args) == 1 and isinstance(args[0], str):
        return reply
    if kind == "bye" and len(args) == 1 and (args[0] is None or _non_negative(args[0])):
        return reply
    raise _BadReply(f"unexpected reply {_short_repr(reply, 100)}")


_PROTECTIONS = {   # what each name from _apply_limits() protects, in words for the warning below
    "setsid": "killing the processes an agent starts",
    "rlimits": "the memory, CPU-time, file-size and open-file limits",
    "RLIMIT_AS": "the memory limit",
    "RLIMIT_CPU": "the CPU-time limit",
    "RLIMIT_FSIZE": "the file-size limit",
    "RLIMIT_NOFILE": "the open-file limit",
}
_reported_missing: set[str] = set()   # protections this process has already warned about


def _warn_missing(missing: list[str]) -> None:
    """Print, once per process and protection, which protections the agent process lacks.

    missing: the names _apply_limits() returned in the agent process, e.g. ["RLIMIT_AS"].
    The games still run; they only lack those protections. The tournament runs on Linux,
    where all of them apply.
    """
    new = [name for name in missing if name not in _reported_missing]
    if new:
        _reported_missing.update(new)
        what = "; ".join(_PROTECTIONS.get(name, name) for name in new)
        print(f"WARNING: not available on this platform: {what}. The games still run; the "
              "tournament machine (Linux) enforces every limit.", file=sys.stderr)


class SubprocessRunner:
    """Runs the agent in its own child process, so it cannot freeze or damage the referee.

    The child runs agent_host_main (above) under limits the operating system enforces: a step()
    that runs past the hard limit is killed, memory and CPU time are capped, the agent works in an
    empty temporary folder, its prints go to a size-capped log file, and imports outside the
    allow-list fail. Used by the tournament, the validator and `simulator.py --strict`.
    Starting the child costs about 0.3 s per agent per game.
    agent_spec, registry_name: as for InProcessRunner.
    limits: replacements for some default limits (keys: memory_limit_mb, hard_limit, max_plies,
            init_limit, log_cap_bytes); None keeps every default from constants.
    log_dir: the folder for the agent's log file; None makes the log a temporary file of its own.
             Either way the log is kept after stop(); the agent's temporary folder is not.
    """

    def __init__(self, agent_spec: str, registry_name: str = "student_agent", limits: dict | None = None,
                 log_dir: str | None = None):
        import constants
        if _is_file_spec(agent_spec):
            agent_spec = os.path.abspath(agent_spec)   # the child changes directory before it loads the file
        self.agent_spec, self.registry_name = agent_spec, registry_name
        self.name = agent_display_name(agent_spec)
        self.limits = {"memory_limit_mb": constants.MEMORY_LIMIT_MB, "hard_limit": constants.STEP_HARD_LIMIT,
                       "max_plies": constants.MAX_PLIES, "init_limit": constants.INIT_TIME_LIMIT,
                       "log_cap_bytes": constants.LOG_CAP_BYTES}
        self.limits.update(limits or {})
        self.log_dir = log_dir
        self.proc = self.conn = None                   # the child process and our end of the pipe; set by start()
        self.work_dir = self.log_path = None           # the private folder and the log file; set by start()
        self.init_time, self.max_rss_mb = 0.0, None    # seconds in __init__, peak memory in MB; see stats()

    def _kill_group(self) -> None:
        """Kill the agent process and everything it started, at once and without appeal.

        SIGKILL cannot be caught or ignored, and it goes to the whole process group the child
        made with setsid(), so a subprocess the agent spawned dies with it. Where process groups
        are not available only the child itself is killed. Safe to call on a process that is
        already gone: once a process has been reaped, its id may belong to another process,
        so it is never signalled again.
        """
        if self.proc is None or self.proc.pid is None:   # never started
            return
        import signal
        try:
            if hasattr(os, "killpg"):
                if hasattr(os, "waitid"):
                    try:   # WNOWAIT: look at the child's state without reaping it
                        os.waitid(os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    except ChildProcessError:
                        return   # already reaped
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.kill()
            except Exception:
                pass

    def _death_report(self) -> str:
        """Describe how the agent process ended: the end of its log, then its exit code or signal.

        Waits up to one second for the process to end, kills its group (so anything the agent
        started dies too), then reaps it and reads the exit code. The exit line comes last
        because the game record keeps only the last 2 000 characters.
        Example: "Traceback ... MemoryError\nthe agent process exited with code 1"
        """
        import signal
        from multiprocessing.connection import wait
        ended = bool(wait([self.proc.sentinel], timeout=1.0))   # ready once the process has exited
        self._kill_group()
        self.proc.join(1.0)
        code = self.proc.exitcode
        if not ended:
            ending = "the agent process broke its connection to the referee and was killed"
        elif code is None:
            ending = "the agent process did not end"
        elif code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = str(-code)
            hint = {"SIGXCPU": " (the CPU-time limit)", "SIGSEGV": " (a crash, often from too deep a recursion)"}
            ending = f"the agent process was killed by signal {name}{hint.get(name, '')}"
        else:
            ending = f"the agent process exited with code {code}"
        tail = _tail(self.log_path).rstrip()
        return f"{tail}\n{ending}" if tail else f"{ending}; its log is empty"

    def _receive(self) -> list:
        """Read the next reply of the agent process and check it with _decode_reply().

        Raises EOFError or OSError when the pipe is closed or broken, OSError also for a reply
        longer than _MAX_REPLY_BYTES (it is not read), and _BadReply for a malformed one.
        """
        return _decode_reply(self.conn.recv_bytes(_MAX_REPLY_BYTES))

    def start(self, seat: int, rules: dict) -> None:
        """Start the agent's process, load the agent in it and wait until it reports ready.

        seat : the agent's seat in this game, 1 or 2; only used to name the log file.
        rules: this game's rule flags, e.g. {"allow_moves": True, "win_length": 3}. The child
               sets them in its own constants before importing the agent and echoes them back;
               start() refuses to go on if the echo differs from what was asked.
        Creates the agent's private temporary folder (its working directory) and its log file,
        then starts the child with the "spawn" method: a fresh interpreter that inherits nothing
        from the referee. Sets self.init_time to the seconds the agent's __init__ took, or to
        init_limit after a timeout. When anything fails, stop() runs before the exception is
        raised, so no process and no temporary folder is left behind, also after Ctrl-C.
        Raises AgentInitTimeout: no "ready" within init_limit seconds.
               AgentLoadError  : the file failed to import, __init__ raised, the child died while
                                 starting (the end of its log is in the message), the echoed
                                 rule flags differ from `rules`, or the child's reply was malformed.
        """
        import multiprocessing
        import tempfile
        try:
            self.work_dir = tempfile.mkdtemp(prefix="agent_")
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                self.log_path = os.path.join(self.log_dir, f"seat{seat}_{uuid.uuid4().hex[:8]}.txt")
            else:
                # Next to the work folder, not inside it, so that stop() can always delete the folder.
                fd, self.log_path = tempfile.mkstemp(prefix=f"agent_seat{seat}_", suffix=".txt")
                os.close(fd)
            cfg = dict(self.limits, agent_file=self.agent_spec, registry_name=self.registry_name,
                       rules=dict(rules), log_path=self.log_path, work_dir=self.work_dir)
            ctx = multiprocessing.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=agent_host_main, args=(child, cfg), daemon=True)
            # One BLAS thread per agent process: in a parallel tournament the agents must not each
            # start a thread for every core. The child inherits the variables when it starts; they
            # are removed again afterwards, so the referee's own environment is left as it was.
            added = [var for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                     if var not in os.environ]
            os.environ.update(dict.fromkeys(added, "1"))
            try:
                self.proc.start()
            finally:
                for var in added:
                    os.environ.pop(var, None)
            child.close()   # our copy of the child's end; while it stays open, a dead child never shows as EOF
            if not self.conn.poll(self.limits["init_limit"]):
                self._kill_group()
                self.init_time = self.limits["init_limit"]
                raise AgentInitTimeout(f"{self.name}: not ready after {self.limits['init_limit']} s")
            try:
                reply = self._receive()
            except (EOFError, OSError):
                raise AgentLoadError(f"{self.name}: agent process died during start-up\n{self._death_report()}") from None
            except _BadReply as exc:
                raise AgentLoadError(f"{self.name}: the agent process broke the protocol: {exc}") from None
            if reply[0] == "load_error":
                raise AgentLoadError(f"{self.name}: {reply[1]}")
            if reply[0] != "ready":
                raise AgentLoadError(f"{self.name}: the agent process broke the protocol: unexpected reply {reply[0]}")
            self.init_time = reply[1]
            if reply[2] != dict(rules):
                raise AgentLoadError(f"{self.name}: rule flags mismatch, asked {rules}, agent process has {reply[2]}")
            _warn_missing(reply[3])
        except BaseException:
            self.stop()
            raise

    def step(self, snapshot, player, opponent, hard_limit) -> StepResult:
        """Ask the agent process for one move and report what happened, within hard_limit seconds.

        snapshot  : (board, reserves); copies, sent to the child through the pipe, so the agent
                    can never touch the referee's own state.
        player    : the agent's player number, 1 or 2. opponent: the other number.
        hard_limit: seconds to wait for the reply. Past it the whole process group is killed.
        Returns one of four StepResults:
          "ok"           the agent returned a value, not checked here; agent_time is the time
                         inside step() as measured by the child, wall_time the time waited here.
          "error"        step() raised; error holds the traceback.
          "hard_timeout" no reply in time; the agent is dead and agent_time is set to hard_limit.
          "crash"        the pipe broke, so the process died (os._exit, out of memory, a signal),
                         or its reply was malformed; error holds the end of the agent's log and
                         how the process ended.
        Raises RuntimeError if called before start() or after stop().
        """
        if self.proc is None:   # the caller's bug, not the agent's: no forfeit
            raise RuntimeError(f"{self.name}: step() called before start() or after stop()")
        board, reserves = snapshot
        t0 = time.perf_counter()
        try:
            self.conn.send(("step", board, reserves, player, opponent))
            if not self.conn.poll(hard_limit):   # poll first: recv() alone would wait forever on a hung agent
                self._kill_group()
                return StepResult("hard_timeout", None, hard_limit, time.perf_counter() - t0, None)
            reply = self._receive()
        except (EOFError, OSError):   # the pipe is closed or broken (BrokenPipeError is an OSError)
            wall = time.perf_counter() - t0
            return StepResult("crash", None, 0.0, wall, self._death_report())
        except _BadReply as exc:
            self._kill_group()
            return StepResult("crash", None, 0.0, time.perf_counter() - t0, f"the agent process broke the protocol: {exc}")
        wall = time.perf_counter() - t0
        if reply[0] == "ok":
            # The time inside step() cannot exceed the time the referee waited for the answer.
            return StepResult("ok", reply[1], min(reply[2], wall), wall, None)
        if reply[0] == "error":
            return StepResult("error", None, min(reply[1], wall), wall, reply[2])
        self._kill_group()
        return StepResult("crash", None, 0.0, wall, f"the agent process broke the protocol: unexpected reply {reply[0]}")

    def _has_exited(self) -> bool:
        """True if the agent process has ended. Unlike Process.is_alive(), this never reaps it."""
        from multiprocessing.connection import wait
        return bool(wait([self.proc.sentinel], timeout=0))

    def stop(self) -> None:
        """End the agent's process and clean up after it. Safe to call twice, and after a failed start().

        A polite ("quit",) goes first, with one second for the child to answer ["bye", max_rss_mb]
        so that the agent's peak memory is recorded. Then the whole process group is killed
        regardless, so a child that ignored the request, or is hung, dies too, and the pipe is
        closed. Afterwards the agent's temporary folder is removed with everything the agent wrote
        in it, and the log is trimmed to log_cap_bytes. The log is kept: stats() names it.
        Leaves self.proc as None.
        """
        if self.proc is not None:
            try:
                if self.proc.pid is not None and not self._has_exited():
                    self.conn.send(("quit",))
                    if self.conn.poll(1.0):
                        reply = self._receive()
                        if reply[0] == "bye":
                            self.max_rss_mb = reply[1]
            except (EOFError, OSError, _BadReply):
                pass   # the child is already dead or broken; its memory figure is lost, nothing else to do
            finally:   # also after Ctrl-C: the process must not outlive the game
                self._kill_group()
                if self.proc.pid is not None:   # join() refuses a process that was never started
                    self.proc.join(2.0)
                self.conn.close()
                self.proc = self.conn = None
        if self.work_dir:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
        try:
            if self.log_path and os.path.getsize(self.log_path) > self.limits["log_cap_bytes"]:
                with open(self.log_path, "rb+") as f:
                    f.truncate(self.limits["log_cap_bytes"])
        except OSError:
            pass   # no log, or it cannot be trimmed; clean-up must never turn a finished game into an error

    def stats(self) -> dict:
        """Return {"init_time": seconds, "max_rss_mb": MB or None, "log_path": path}.

        max_rss_mb is the agent process's peak memory, known only after stop() and only when the
        child answered the quit message. log_path is the agent's log file, which outlives stop().
        """
        return {"init_time": self.init_time, "max_rss_mb": self.max_rss_mb, "log_path": self.log_path}
