"""Validate one student_agent.py before it enters the tournament.

Usage: python validate_submission.py path/to/student_agent.py [options]; -h lists the options.

Static phase: the file is read and parsed, never executed. Dynamic phase: the file is loaded
in a sandboxed child process and plays a few games against random_agent. Every check adds one
[PASS], [WARN] or [FAIL] line to the report; a single FAIL makes the verdict FAIL.
Exit code 0 = PASS (possibly with warnings), 1 = FAIL, 2 = the validator itself failed.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import platform
import re
import sys
import tempfile
import tokenize
import traceback
from dataclasses import dataclass, field

import numpy as np

import constants
from allowed_imports import is_allowed

MAX_FILE_BYTES = 2_000_000          # 2 MB; a larger file is a FAIL (C-01)
GRADING_PYTHON = "3.11"             # the Python and numpy versions of the tournament machine (C-02)
GRADING_NUMPY = "1.26"
# Built-ins an agent must not call, unless the file defines a function of that name (an eval of its own).
FORBIDDEN_BUILTINS = {"exit", "quit", "exec", "eval", "compile", "breakpoint", "input"}
# Calls that end the agent's own process instead of returning a move.
EXIT_CALLS = {"sys.exit", "os._exit", "os.abort"}
# Calls that work with files, besides the built-in open(); only a WARN.
FILE_CALLS = {"io.open", "os.open", "os.fdopen", "os.listdir", "os.scandir", "os.remove", "os.unlink", "os.rename",
              "os.replace", "os.mkdir", "os.makedirs", "os.rmdir", "pickle.load", "pickle.dump", "json.load",
              "json.dump", "numpy.load", "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.loadtxt",
              "numpy.savetxt", "numpy.genfromtxt", "numpy.fromfile", "numpy.memmap"}
FILE_MODULES = {"pathlib", "shutil", "tempfile", "glob"}   # modules whose every call works with files
# Modules an agent must not alter: assigning to their attributes, or setattr() on them, is monkey-patching.
PATCHABLE_MODULES = {"helpers", "constants", "numpy", "store"}
# The rule flags. An agent reads them from constants and must never assign them (C-13).
FLAG_NAMES = {"ALLOW_MOVES", "WIN_LENGTH"}
# Game-end reasons that mean the agent itself broke.
FORFEIT_REASONS = ("hard_timeout", "crash", "too_many_strikes", "init_timeout")
# Move flags that are notes, not strikes: a tuple is accepted as a move, with a warning.
NOTE_FLAGS = {"tuple_move"}
# A time that reaches this share of its limit is a WARN: the tournament machine may be slower.
TIME_MARGIN = 0.9
# Calls allowed on the right-hand side of a module-level constant, e.g. TABLE = np.zeros((4, 4)).
SIMPLE_CALLS = {"array", "dict", "set", "tuple", "list", "frozenset", "zeros", "ones", "full", "arange",
                "float", "int", "bool", "str", "range"}


@dataclass
class Finding:
    """One line of the report: the result of one check.

    Example: Finding("WARN", "print() inside step()", 12).text() -> "[WARN] line 12: print() inside step()"
    """
    level: str              # "PASS" | "WARN" | "FAIL"
    message: str            # what was checked, or what is wrong, in one line
    line: int | None = None # the line of the file the finding points at; None when it is about the whole file
    detail: list = field(default_factory=list)   # more lines shown under it, e.g. the first error of each kind

    def text(self) -> str:
        """Return the finding as one report line, e.g. "[FAIL] line 3: syntax error: expected ':'"."""
        where = f"line {self.line}: " if self.line else ""
        return f"[{self.level}] {where}{self.message}"


@dataclass
class Report:
    """Everything the validator found out about one file.

    The static and dynamic phases add findings to it. main() prints it with format_report and,
    with --json, writes to_dict() to a file that the tournament reads.
    """
    path: str                                        # the validated file, as an absolute path
    findings: list = field(default_factory=list)     # Finding objects, in the order the checks ran
    dynamic: list = field(default_factory=list)      # one dict per rule set played, from dynamic_phase
    class_name: str | None = None                    # the registered class, e.g. "StudentAgent"; None until found

    def add(self, level: str, message: str, line: int | None = None, detail: list | None = None) -> None:
        """Append one finding. level is "PASS", "WARN" or "FAIL"; line and detail are optional."""
        self.findings.append(Finding(level, message, line, list(detail or [])))

    @property
    def verdict(self) -> str:
        """Return "FAIL" if any finding is a FAIL, else "PASS". Warnings do not change it."""
        return "FAIL" if any(f.level == "FAIL" for f in self.findings) else "PASS"

    @property
    def warnings(self) -> int:
        """Return the number of WARN findings."""
        return sum(1 for f in self.findings if f.level == "WARN")

    def to_dict(self) -> dict:
        """Return the report as plain dicts, lists, strings and numbers, ready for json.dumps.

        The interpreter and numpy versions are included, so a saved report shows which Python
        checked the file.
        """
        return {"path": self.path, "verdict": self.verdict, "warnings": self.warnings,
                "findings": [vars(f) for f in self.findings], "dynamic": self.dynamic,
                "class_name": self.class_name, "python": platform.python_version(), "numpy": np.__version__}


def check_versions(report: Report, python: str = platform.python_version(),
                   numpy_version: str = np.__version__) -> None:
    """Warn when this check runs on other Python or numpy versions than the tournament (C-02).

    Takes: report        - receives a WARN on a mismatch, nothing otherwise.
           python        - the interpreter's version, e.g. "3.12.1".
           numpy_version - numpy's version, e.g. "2.0.0".
    A newer interpreter accepts syntax that the tournament's rejects, and numpy 2 changed functions.
    """
    if python.rsplit(".", 1)[0] != GRADING_PYTHON or numpy_version.rsplit(".", 1)[0] != GRADING_NUMPY:
        report.add("WARN", f"this check runs on Python {python} with numpy {numpy_version}, but the tournament uses "
                           f"Python {GRADING_PYTHON} with numpy {GRADING_NUMPY}; run it with those versions (C-02)")


def read_source(path: str, report: Report) -> str | None:
    """Check the file's name, size and encoding (C-01).

    Takes:   path   - the submission file.
             report - receives one finding per check.
    Returns: the file's text, or None after a FAIL: the file cannot be read, is larger than
             MAX_FILE_BYTES, contains NUL bytes, or cannot be decoded. A wrong file name, a
             byte-order mark or a coding line other than UTF-8 is only a WARN: Python reads such
             a file, and the file is renamed when it is collected.
    """
    name_ok = os.path.basename(path) == "student_agent.py"
    if not name_ok:
        report.add("WARN", f"file is named {os.path.basename(path)!r}; the submission must be student_agent.py (C-01)")
    # Raw bytes for the NUL and encoding checks; one byte past the limit is enough to know it is too big.
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            raw = f.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        report.add("FAIL", f"cannot read the file: {exc}")
        return None
    if len(raw) > MAX_FILE_BYTES:
        report.add("FAIL", f"file is {max(size, len(raw))} bytes, limit {MAX_FILE_BYTES} (C-01)")
        return None
    if b"\x00" in raw:
        report.add("FAIL", "file contains NUL bytes; is it UTF-16? (C-01)", raw[:raw.index(b"\x00")].count(b"\n") + 1)
        return None
    try:
        # The encoding Python itself uses for a source file: UTF-8, unless a BOM or a coding line says otherwise.
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        source = raw.decode(encoding)
    except (SyntaxError, LookupError) as exc:     # a coding line that names no known encoding
        report.add("FAIL", f"the file's coding line cannot be used: {exc} (C-01)")
        return None
    except UnicodeDecodeError as exc:
        line = raw[:exc.start].count(b"\n") + 1
        name = "UTF-8" if encoding.startswith("utf-8") else encoding
        report.add("FAIL", f"file is not {name}: byte {raw[exc.start]:#04x} cannot be decoded (C-01)", line)
        return None
    if encoding == "utf-8-sig":
        report.add("WARN", "file starts with a UTF-8 byte-order mark; Python accepts it, but save the file as UTF-8 "
                           "without one (C-01)")
    elif encoding != "utf-8":
        report.add("WARN", f"file is encoded as {encoding}, not UTF-8; Python accepts it because of its coding line "
                           "(C-01)")
    checked = ["name"] * name_ok + ["size"] + ["encoding"] * (encoding == "utf-8")
    report.add("PASS", "file " + " / ".join(checked))
    return source


def parse_source(source: str, report: Report) -> ast.Module | None:
    """Check that the file parses under this Python version (C-02).

    Takes:   source - the file's text.
             report - receives the finding.
    Returns: the ast.Module, or None after a FAIL: a syntax error, with its line, or a file
             nested too deeply for Python to compile at all. A newer interpreter parses with the
             grammar of GRADING_PYTHON, so syntax the tournament rejects is a FAIL here too.
    """
    try:
        tree = ast.parse(source, feature_version=tuple(int(n) for n in GRADING_PYTHON.split(".")))
    except SyntaxError as exc:
        report.add("FAIL", f"syntax error: {exc.msg} (C-02)", exc.lineno)
        return None
    except (RecursionError, MemoryError):
        # Python's own compiler fails on such a file too, so it could never be imported.
        report.add("FAIL", "the file is nested too deeply for Python to compile (C-02)")
        return None
    report.add("PASS", "syntax")
    return tree


def _dotted_name(node) -> str:
    """Return the dotted name that an expression spells, e.g. "np.random" for np.random.

    Takes:   node - any expression node.
    Returns: the names joined by dots; "" when the expression does not start with a plain name,
             e.g. f().x or table[0].
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(node: ast.Call) -> str:
    """Return the dotted name of what a call node calls.

    Takes:   node - an ast.Call.
    Returns: "open" for open(...), "sys.exit" for sys.exit(...), "np.random.seed" for the
             dotted form; "" when the callee does not start with a plain name, e.g. f()(x),
             table[0](x) or super().__init__().
    """
    return _dotted_name(node.func)


def _import_aliases(tree: ast.Module) -> dict:
    """Map each name that an import binds to the dotted name it stands for.

    Takes:   tree - the parsed file.
    Returns: e.g. {"np": "numpy", "s": "sys", "leave": "sys.exit"} for `import numpy as np`,
             `import sys as s` and `from sys import exit as leave`. A plain `import sys` needs
             no entry, and `from constants import *` binds nothing that can be listed.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname: a.name for a in node.names if a.asname})
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names if a.name != "*"})
    return aliases


def _resolve(name: str, aliases: dict) -> str:
    """Return a dotted name with its first part replaced by what the file imported under it.

    Example: with aliases {"np": "numpy"}, "np.random.seed" -> "numpy.random.seed". A name the
    imports do not bind, e.g. "self.board", is returned unchanged.
    """
    head, dot, rest = name.partition(".")
    return aliases.get(head, head) + dot + rest


def _is_main_guard(node) -> bool:
    """Return True for exactly `if __name__ == "__main__":`, in either order.

    The framework loads a submission under a module name of its own, so the body of this block
    never runs in a game; only the student runs it, with `python student_agent.py`.
    """
    test = node.test if isinstance(node, ast.If) else None
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = (test.left, test.comparators[0])
    return any(isinstance(n, ast.Name) and n.id == "__name__" for n in sides) and \
        any(isinstance(n, ast.Constant) and n.value == "__main__" for n in sides)


def _code_nodes(tree: ast.Module) -> list:
    """Return the nodes of every statement and expression that can run in a game, in file order.

    Takes:   tree - the parsed file.
    Returns: the nodes sorted by line and column (ast.walk visits a tree level by level), without
             the body of a module-level `if __name__ == "__main__":` block.
    """
    skipped = {id(n) for top in tree.body if _is_main_guard(top) for stmt in top.body for n in ast.walk(stmt)}
    nodes = [n for n in ast.walk(tree) if hasattr(n, "lineno") and id(n) not in skipped]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def scan_imports(tree: ast.Module, report: Report) -> None:
    """Check that every import that can run is allowed: numpy and the standard library (C-03).

    Takes: tree   - the parsed file.
           report - receives one FAIL per import that is not allowed, with its line, or one
                    PASS that lists every imported module once, in file order.
    Imports inside functions count too. `from agents import x` is checked as agents.x, and
    __import__("x") or importlib.import_module("x") as an import of x. A relative import is a FAIL.
    """
    aliases = _import_aliases(tree)
    seen, bad = [], False
    for node in _code_nodes(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                report.add("FAIL", "relative imports are not allowed (C-03)", node.lineno); bad = True
                continue
            names = [node.module]
            if node.module == "agents":    # `from agents import random_agent` imports agents.random_agent
                names += [f"agents.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call) and \
                _resolve(_call_name(node), aliases) in ("__import__", "importlib.import_module"):
            arg = node.args[0] if node.args else None
            # A computed module name cannot be checked here; the sandbox checks it when it runs.
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names = [arg.value]
        for name in names:
            seen.append(name)
            if not is_allowed(name):
                report.add("FAIL", f"import of {name!r} is not allowed (C-03)", node.lineno); bad = True
    if not bad:
        report.add("PASS", "imports: " + (", ".join(dict.fromkeys(seen)) or "none"))


def _registration_name(decorator) -> str | None:
    """Return the name a decorator registers, if it is a register_agent call.

    Takes:   decorator - one node from a ClassDef's decorator_list.
    Returns: the string inside @register_agent("..."), @register_agent(agent_name="...") or
             @store.register_agent("..."); "" for a register_agent call whose argument is not one
             plain string; None for any other decorator.
    """
    if isinstance(decorator, ast.Call) and _call_name(decorator).endswith("register_agent"):
        values = decorator.args + [k.value for k in decorator.keywords if k.arg == "agent_name"]
        if len(values) == 1 and isinstance(values[0], ast.Constant) and isinstance(values[0].value, str):
            return values[0].value
        return ""
    return None


def scan_registration(tree: ast.Module, report: Report) -> ast.ClassDef | None:
    """Check that exactly one class is decorated @register_agent("student_agent") (C-04).

    Takes:   tree   - the parsed file.
             report - receives a FAIL (no registration, "student_agent" registered twice, or a
                      registration under another name) or a PASS. On PASS, report.class_name is
                      set to the name of the class.
    Returns: the ClassDef of the registered class, or None after a FAIL.
    """
    registered = []                                       # (class node, registered name), one per decorator
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names = [n for n in (_registration_name(d) for d in node.decorator_list) if n is not None]
            if len(names) > 1:
                report.add("FAIL", f"class {node.name} is decorated with register_agent more than once (C-04)",
                           node.lineno)
            for name in names:
                registered.append((node, name))
    good = [(c, n) for c, n in registered if n == "student_agent"]
    if not registered:
        report.add("FAIL", 'no class is decorated with @register_agent("student_agent") (C-04)')
        return None
    if len(good) > 1:
        report.add("FAIL", 'more than one class is registered as "student_agent" (C-04)', good[1][0].lineno)
        return None
    if not good:
        cls, name = registered[0]
        found = f"as {name!r}" if name else "with a name that is not one plain string"
        report.add("FAIL", f'class {cls.name} is registered {found}; it must be "student_agent" (C-04)', cls.lineno)
        return None
    cls = good[0][0]
    report.class_name = cls.name
    report.add("PASS", f'registration: class {cls.name} registered as "student_agent"')
    return cls


def _class_chain(tree: ast.Module, cls: ast.ClassDef) -> list:
    """Return cls, then the base classes that the same file defines for it, nearest first.

    Example: with `class Searcher(Agent)` and `class StudentAgent(Searcher)` in one file, the
    chain of StudentAgent is [StudentAgent, Searcher]. Each class appears once.
    """
    local = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    chain, todo = [], [cls]
    while todo:
        c = todo.pop(0)
        if c not in chain:
            chain.append(c)
            todo += [local[b.id] for b in c.bases if isinstance(b, ast.Name) and b.id in local]
    return chain


def _find_method(chain: list, name: str):
    """Return the definition of method `name` that a class chain uses, or None if it has none."""
    for c in chain:
        # reversed: when a class body defines a method twice, the last definition is the one used.
        for node in reversed(c.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
    return None


def _required_parameters(fn) -> tuple:
    """Return the names of a function's parameters that have no default: (positional, keyword-only).

    Example: for def step(self, board, depth=3, *, verbose): -> (["self", "board"], ["verbose"]).
    """
    positional = fn.args.posonlyargs + fn.args.args
    required = positional[:len(positional) - len(fn.args.defaults)]   # defaults belong to the last ones
    keyword = [a.arg for a, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if default is None]
    return [a.arg for a in required], keyword


def check_class(tree: ast.Module, cls: ast.ClassDef, report: Report) -> None:
    """Check that the framework can create the agent and call its step (C-05, C-06).

    Takes: tree   - the parsed file; the base classes it defines are searched for methods too.
           cls    - the ClassDef that scan_registration returned.
           report - receives the findings. WARN: Agent is not a base class, or __init__ does
                    not call super().__init__(). FAIL: __init__ needs an argument; step is missing,
                    async, or cannot be called as step(board, reserves, player, opponent).
    """
    chain = _class_chain(tree, cls)
    # A base is a Name (Agent) or an Attribute (agents.agent.Agent); only the last part is compared.
    bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for c in chain for b in c.bases]
    if "Agent" not in bases:
        report.add("WARN", f"class {cls.name} does not list Agent as a base class (C-05)", cls.lineno)
    init = _find_method(chain, "__init__")
    if init is not None:
        positional, keyword = _required_parameters(init)
        if positional[1:] or keyword:                  # positional[0] is self
            report.add("FAIL", f"__init__ must take no arguments besides self, found {positional[1:] + keyword} (C-05)",
                       init.lineno)
        calls_super = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__init__"
                          for n in ast.walk(init))
        if not calls_super:
            report.add("WARN", "__init__ does not call super().__init__() (C-05)", init.lineno)
    step = _find_method(chain, "step")
    if step is None:
        report.add("FAIL", f"class {cls.name} has no step method (C-06)", cls.lineno)
        return
    if isinstance(step, ast.AsyncFunctionDef):
        report.add("FAIL", "step must not be async (C-06)", step.lineno)
    positional, keyword = _required_parameters(step)
    n_pos = len(step.args.posonlyargs + step.args.args)
    wanted = 4 if any(_dotted_name(d) == "staticmethod" for d in step.decorator_list) else 5   # self counts
    if (n_pos < wanted and not step.args.vararg) or len(positional) > wanted or keyword:
        report.add("FAIL", "step must take (self, board, reserves, player, opponent) and give any other parameter "
                           f"a default, found step({ast.unparse(step.args)}) (C-06)", step.lineno)
    else:
        report.add("PASS", "step(self, board, reserves, player, opponent)")


def _assigned_targets(node) -> list:
    """Return the names and attributes that one node assigns to.

    Takes:   node - any node; assignments, for loops and `:=` expressions assign something.
    Returns: the target nodes in order, with tuples and lists unpacked, e.g. the names
             ALLOW_MOVES and WIN_LENGTH for `ALLOW_MOVES, WIN_LENGTH = True, 3`; [] otherwise.
    """
    if isinstance(node, ast.Assign):
        todo = list(node.targets)                     # a = b = 1 has two targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
        todo = [node.target]
    else:
        return []
    found = []
    while todo:
        target = todo.pop(0)
        if isinstance(target, (ast.Tuple, ast.List)):
            todo[:0] = target.elts
        elif isinstance(target, ast.Starred):
            todo.insert(0, target.value)
        else:
            found.append(target)
    return found


def _enclosing_functions(tree: ast.Module) -> dict:
    """Map the id of every node inside a function to the innermost function that contains it.

    Nodes at module or class level are not in the map. ast.walk reaches an outer function before
    the functions inside it, so an inner function overwrites the entries of the outer one.
    """
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            owner.update({id(n): fn for n in ast.walk(fn) if n is not fn})
    return owner


def _defined_names(tree: ast.Module) -> set:
    """Return every name the file binds itself: functions, classes, assignments, parameters, imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        names.update(t.id for t in _assigned_targets(node) if isinstance(t, ast.Name))
    return names


def _int_value(node, depth: int = 0) -> int | None:
    """Return the value of an integer written with literals only, e.g. 10_000, 10 ** 4 or int(1e4).

    Returns None for anything else, and for a value too large or an expression too deep to
    evaluate cheaply.
    """
    if depth > 20:
        return None
    value = None
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
    elif isinstance(node, ast.Call) and _call_name(node) == "int" and len(node.args) == 1 and not node.keywords \
            and isinstance(node.args[0], ast.Constant) and type(node.args[0].value) in (int, float):
        value = int(node.args[0].value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _int_value(node.operand, depth + 1)
        value = None if operand is None else -operand
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        left, right = _int_value(node.left, depth + 1), _int_value(node.right, depth + 1)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif 0 <= right <= 64:
                value = left ** right
    return value if value is not None and abs(value) <= 10 ** 18 else None


def _check_call(node: ast.Call, aliases: dict, defined: set, report: Report) -> None:
    """Add a finding for one call that is forbidden or deserves a warning.

    FAIL: FORBIDDEN_BUILTINS, EXIT_CALLS, a recursion limit above 10 000 or computed,
    configure_rules(), setattr() or delattr() on a patchable module. WARN: a literal recursion
    limit of at most 10 000, globals(), np.random.seed(), file I/O.
    """
    written = _call_name(node)
    name = _resolve(written, aliases).replace("__builtins__.", "builtins.", 1)
    # A built-in function: called through builtins, or by a plain name that the file never binds.
    if name.startswith("builtins."):
        builtin = name[len("builtins."):]
    else:
        builtin = name if name == written and "." not in name and name not in defined else ""
    if builtin in FORBIDDEN_BUILTINS:
        report.add("FAIL", f"call to {written}() is not allowed (C-10)", node.lineno)
    elif name in EXIT_CALLS:
        report.add("FAIL", f"{written}() is not allowed (C-10)", node.lineno)
    elif name == "sys.setrecursionlimit":
        # A modest literal limit is a WARN. A huge or computed one lets a deep recursion
        # overflow the C stack and kill the process instead of raising RecursionError.
        limit = _int_value(node.args[0]) if node.args else None
        if limit is not None and limit <= 10_000:
            report.add("WARN", f"sys.setrecursionlimit({limit}): allowed up to 10 000; a loop is faster than deep "
                               "recursion (C-10)", node.lineno)
        else:
            report.add("FAIL", "sys.setrecursionlimit above 10 000 or with a value that is not a plain number (C-10)",
                       node.lineno)
    elif builtin == "configure_rules" or name == "constants.configure_rules":
        # A plain configure_rules() comes from `from constants import *`.
        report.add("FAIL", f"call to {written}() is not allowed: agents only read the rule flags (C-13)", node.lineno)
    elif builtin in ("setattr", "delattr") and node.args \
            and _resolve(_dotted_name(node.args[0]), aliases).split(".")[0] in PATCHABLE_MODULES:
        report.add("FAIL", f"{builtin} on {_dotted_name(node.args[0])} is monkey-patching (C-10)", node.lineno)
    elif builtin == "globals":
        report.add("WARN", "globals() is rarely needed in an agent; make sure it does not change module state",
                   node.lineno)
    elif name == "numpy.random.seed":
        report.add("WARN", "np.random.seed changes the global generator; use np.random.default_rng()", node.lineno)
    elif builtin == "open" or name in FILE_CALLS or (name.split(".")[0] in FILE_MODULES and "." in name):
        report.add("WARN", f"{written}() works with files, which the tournament rules forbid (C-10); "
                           "the agent's folder is empty and is deleted after every game", node.lineno)


def _check_target(node, target, aliases: dict, owner: dict, report: Report) -> None:
    """Add a finding for one assignment target that sets a rule flag or alters a patchable module.

    FAIL: a rule flag assigned outside a function, an attribute of a patchable module. WARN: a
    local variable named like a rule flag.
    """
    if isinstance(target, ast.Name) and target.id in FLAG_NAMES:
        fn = owner.get(id(node))
        # Inside a function the name is a local variable, unless the function declares it global.
        if fn is None or any(isinstance(n, ast.Global) and target.id in n.names for n in ast.walk(fn)):
            report.add("FAIL", f"assignment to {target.id}: the rule flags are read-only for agents (C-13)",
                       node.lineno)
        else:
            report.add("WARN", f"a local variable named {target.id} hides the rule flag; "
                               f"read constants.{target.id} instead (C-13)", node.lineno)
    elif isinstance(target, ast.Attribute):
        base = _dotted_name(target.value)
        if _resolve(base, aliases).split(".")[0] in PATCHABLE_MODULES:
            report.add("FAIL", f"assignment to {base}.{target.attr} is monkey-patching (C-10, C-13)", node.lineno)


def scan_forbidden(tree: ast.Module, report: Report) -> None:
    """Check for calls and assignments that break the contract (C-10, C-13); point out file I/O.

    Takes: tree   - the parsed file.
           report - receives one finding per hit, with its line. Names are resolved through the
                    file's imports, so `import sys as s; s.exit()` counts as sys.exit(). FAIL and
                    WARN cases: see _check_call and _check_target. A clean file gets no finding.
    """
    aliases, defined, owner = _import_aliases(tree), _defined_names(tree), _enclosing_functions(tree)
    for node in _code_nodes(tree):
        if isinstance(node, ast.Call):
            _check_call(node, aliases, defined, report)
        for target in _assigned_targets(node):
            _check_target(node, target, aliases, owner, report)


def _is_simple_value(node) -> bool:
    """Return True if an expression is simple enough to be a module-level constant (C-09).

    Takes: node - the right-hand side of a module-level assignment.
    Allowed: literals, names and dotted names such as math.inf; tuples, lists, sets and dicts of
    allowed values; operations and comparisons on them; calls whose last name part is in
    SIMPLE_CALLS, e.g. np.zeros((4, 4)), with allowed arguments. Anything else may run the
    student's own code.
    """
    todo = [node]                  # a list, not recursion: 1 + 1 + ... + 1 may have thousands of terms
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.Constant, ast.Name)) or (isinstance(n, ast.Attribute) and _dotted_name(n)):
            continue
        if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
            todo.extend(n.elts)
        elif isinstance(n, ast.Dict):
            todo.extend(e for e in n.keys + n.values if e is not None)   # a None key is a ** entry
        elif isinstance(n, ast.UnaryOp):
            todo.append(n.operand)
        elif isinstance(n, ast.BinOp):
            todo += [n.left, n.right]
        elif isinstance(n, ast.Compare):
            todo += [n.left] + n.comparators
        elif isinstance(n, ast.BoolOp):
            todo.extend(n.values)
        elif isinstance(n, ast.Call) and _call_name(n).split(".")[-1] in SIMPLE_CALLS:
            todo += n.args + [k.value for k in n.keywords]
        else:
            return False
    return True


def _endless_loop(statement):
    """Return a while loop in a statement that runs at import and can never finish, or None.

    Only the plain case counts: a constant true test such as `while True:`, no break or call in
    the body, no try around it. Such a loop runs forever or raises, so the file can never load;
    any other loop is left to the start-up time limit. Functions defined inside are ignored.
    """
    inner = {id(m) for f in ast.walk(statement) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
             for m in ast.walk(f)}
    nodes = [n for n in ast.walk(statement) if id(n) not in inner]
    if any(isinstance(n, ast.Try) or type(n).__name__ == "TryStar" for n in nodes):   # TryStar is except*
        return None
    for n in nodes:
        if isinstance(n, ast.While) and isinstance(n.test, ast.Constant) and n.test.value \
                and not any(isinstance(m, (ast.Break, ast.Call)) for m in ast.walk(n)):
            return n
    return None


def _statement_name(node) -> str:
    """Return the kind of a statement in plain words for the report, e.g. "for loop"."""
    if isinstance(node, ast.Expr):
        return "call" if isinstance(node.value, ast.Call) else "expression"
    names = {"For": "for loop", "AsyncFor": "for loop", "While": "while loop", "With": "with block",
             "AsyncWith": "with block", "Try": "try block", "TryStar": "try block", "If": "if block",
             "Assert": "assert", "Delete": "del statement", "Raise": "raise", "Global": "global statement",
             "Match": "match block"}
    return names.get(type(node).__name__, type(node).__name__)


def _scan_import_time(statements: list, report: Report, where: str) -> None:
    """Add a finding for each statement of a module or class body that runs code at import.

    Takes: statements - the body of the module or of a class.
           report     - receives the findings; see scan_module_level.
           where      - "module-level", or "class-level" for the body of a class.
    """
    limit = f"{constants.INIT_TIME_LIMIT:g} s start-up limit"
    for node in statements:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
            continue
        if isinstance(node, ast.ClassDef):
            _scan_import_time(node.body, report, "class-level")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                                # a docstring, or any bare string
        if where == "module-level" and _is_main_guard(node):
            _scan_import_time(node.orelse, report, where)
            continue
        if isinstance(node, ast.If) and _is_simple_value(node.test):
            _scan_import_time(node.body + node.orelse, report, where)   # e.g. if TYPE_CHECKING: only what it holds
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and \
                (where == "class-level" or node.value is None or _is_simple_value(node.value)):
            continue                                                # a class attribute, a bare annotation or a constant
        loop = _endless_loop(node)
        if loop is not None:
            report.add("FAIL", f"{where} while loop never ends, so the file can never be loaded (C-09)", loop.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            report.add("WARN", f"{where} assignment computes its value at import time, which counts toward the "
                               f"{limit} (C-09)", node.lineno)
        else:
            report.add("WARN", f"{where} {_statement_name(node)} runs at import time, which counts toward the "
                               f"{limit} (C-09)", node.lineno)


def scan_module_level(tree: ast.Module, report: Report) -> None:
    """Check the code that runs when the file is imported (C-09).

    Takes: tree   - the parsed file.
           report - receives one finding per statement that is not a definition or a simple
                    constant. FAIL: a loop that can never end, e.g. `while True: pass`. WARN: any
                    other code, e.g. a for loop, random.seed(0) or print(); the import counts
                    toward the start-up time limit. PASS when nothing was found.
    Class bodies run at import too and are examined the same way, but any assignment is allowed
    there (a dataclass field). An `if` with a plain test, such as `if TYPE_CHECKING:`, is looked
    into; the body of `if __name__ == "__main__":` is skipped.
    """
    count = len(report.findings)
    _scan_import_time(tree.body, report, "module-level")
    if len(report.findings) == count:
        report.add("PASS", "module level: only imports, constants and definitions")


def scan_hints(tree: ast.Module, report: Report, cls: ast.ClassDef | None) -> None:
    """Point out habits that cost time or output in the tournament. Advice only: never a FAIL.

    Takes: tree   - the parsed file.
           report - receives a WARN for time.time() (perf_counter is steadier), for deepcopy of
                    a board (slow), and for print() inside step() (its output is discarded).
           cls    - the registered class from scan_registration, or None; without it the
                    print() check is skipped. Its step may come from a base class in the file.
    """
    aliases = _import_aliases(tree)
    for node in _code_nodes(tree):
        if isinstance(node, ast.Call):
            name = _resolve(_call_name(node), aliases)
            if name == "time.time":
                report.add("WARN", "time.time() can jump; prefer time.perf_counter() for a move budget", node.lineno)
            elif name in ("copy.deepcopy", "deepcopy"):
                report.add("WARN", "deepcopy of numpy arrays is slow; use board.copy() or helpers.copy_state",
                           node.lineno)
    if cls is not None:
        step = _find_method(_class_chain(tree, cls), "step")
        if step is not None and any(isinstance(n, ast.Call) and _call_name(n) == "print" for n in ast.walk(step)):
            report.add("WARN", "print() inside step() - output is discarded in the tournament", step.lineno)


def static_phase(path: str) -> Report:
    """Run every static check on one file, in the order the report shows them.

    Takes:   path - the file to validate.
    Returns: a Report. The phase stops after a FAIL that makes the rest impossible: an
             unreadable file or a syntax error. The file is never executed.
    """
    report = Report(path=os.path.abspath(path))
    check_versions(report)
    source = read_source(path, report)
    if source is None:
        return report
    tree = parse_source(source, report)
    if tree is None:
        return report
    scan_imports(tree, report)
    cls = scan_registration(tree, report)
    if cls is not None:
        check_class(tree, cls, report)
    scan_forbidden(tree, report)
    scan_module_level(tree, report)
    scan_hints(tree, report, cls)
    return report


def _error_summary(text: str, path: str) -> tuple:
    """Return the last line of a traceback, and the line of the file at `path` where it was raised.

    Example: a ZeroDivisionError in step() -> ("ZeroDivisionError: division by zero", 7). The line
    is None when no frame of the traceback is in that file.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    where = None
    for line in lines:
        match = re.match(r'\s*File "(.*)", line (\d+)', line)
        if match and os.path.abspath(match.group(1)) == path:
            where = int(match.group(2))
    last = lines[-1].strip() if lines else ""
    # A one-line error still carries the referee's prefix, e.g. "P1 team_042/student_agent: ".
    return re.sub(r"^(P[12] \S+: )?(\S+/\S+: )?", "", last), where


def _problem_text(flag: str, move: dict, where: str, path: str) -> str:
    """Describe the first ply that got `flag`, for the report, e.g. "game 1 (seat 1), ply 3: step() raised ..."."""
    if flag == "exception":
        error, line = _error_summary(move.get("detail") or "", path)
        return f"{where}: step() raised {error}" + (f" (line {line})" if line else "")
    if flag == "illegal_move":
        return f"{where}: step() returned {move.get('detail')}, which is not a legal move"
    if flag == "move_not_allowed":
        return f"{where}: step() returned the board move {move.get('detail')}, but board moves are off in this game"
    if flag == "soft_timeout":
        return (f"{where}: step() took {move['agent_time']:.2f} s, more than the {constants.STEP_TIME_LIMIT:g} s "
                "limit, so its move was replaced by a random one")
    if flag == "tuple_move":
        return f"{where}: step() returned a tuple; it was accepted, but return a helpers.Move"
    return f"{where}: {flag}"


def _counts_text(counts: dict) -> str:
    """Return counts as text for a report line, e.g. " (exception 3, soft_timeout 1)"; "" when empty."""
    return f" ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})" if counts else ""


def _play_games(report: Report, games: int, rules: dict, make) -> None:
    """Play the games of dynamic_phase and append their results to report.dynamic.

    Takes: report - the Report; its path is the file to load.
           games  - how many games; see dynamic_phase for the seats and seeds.
           rules  - the rule set of every game.
           make   - a function that turns an agent spec into a new runner.
    Besides the counts, results["examples"] keeps the first occurrence of each problem as one line
    of text. A load error stops the games, since every later game would fail the same way.
    """
    from world import World      # imported here, not at the top: the static phase never needs it
    results = {"rules": dict(rules), "games": [], "W": 0, "D": 0, "L": 0, "max_step": 0.0, "steps": [],
               "strikes": {}, "notes": {}, "forfeits": 0, "passes": 0, "init_time": None, "max_rss_mb": None,
               "load_error": None, "examples": {}}
    examples = results["examples"]                # the first occurrence of each problem, for the report
    for i in range(games):
        # Games 2k and 2k+1 share seed k and swap seats, so the file plays both sides of one position.
        seat = 1 if i % 2 == 0 else 2
        mine = make(report.path)
        other = make("random_agent")
        rec = World(i // 2, mine if seat == 1 else other, other if seat == 1 else mine, **rules).play()
        if rec.load_error:
            results["load_error"] = rec.load_error
            break
        # The slowest start and the highest memory over all games; memory is unknown after a kill.
        results["init_time"] = max(results["init_time"] or 0.0, rec.init_time[seat])
        if rec.max_rss_mb[seat] is not None:
            results["max_rss_mb"] = max(results["max_rss_mb"] or 0.0, rec.max_rss_mb[seat])
        outcome = "D" if rec.winner == 0 else ("W" if rec.winner == seat else "L")
        results[outcome] += 1
        results["forfeits"] += int(rec.winner == 3 - seat and rec.reason in FORFEIT_REASONS)
        recorded = 0
        for m in rec.moves:
            if m.get("player") != seat:           # only this file's plies are measured
                continue
            if m.get("pass"):                     # a pass has no step time and no flags
                results["passes"] += 1
                continue
            results["steps"].append(m["agent_time"])
            for f in m["flags"]:
                kind = "notes" if f in NOTE_FLAGS else "strikes"
                results[kind][f] = results[kind].get(f, 0) + 1
                recorded += kind == "strikes"
                if f not in examples:
                    examples[f] = _problem_text(f, m, f"game {i + 1} (seat {seat}), ply {m['ply'] + 1}", report.path)
        # The referee does not record the ply of the strike that forfeits a game, so its type is
        # unknown; a key of its own keeps the total equal to the referee's.
        if rec.strikes[seat] > recorded:
            unrecorded = rec.strikes[seat] - recorded
            results["strikes"]["forfeiting_strike"] = results["strikes"].get("forfeiting_strike", 0) + unrecorded
        if rec.winner == 3 - seat and rec.reason == "hard_timeout":
            examples.setdefault("hard_timeout", f"game {i + 1} (seat {seat}): step() did not return within "
                                                f"{constants.STEP_HARD_LIMIT:g} s, so the agent was stopped and lost")
        elif rec.winner == 3 - seat and rec.reason == "crash":
            ending, _ = _error_summary(rec.forfeit_detail or "", report.path)
            examples.setdefault("crash", f"game {i + 1} (seat {seat}): the agent's process ended: {ending}")
        log = rec.log_paths[seat]
        results["games"].append({"seed": rec.seed, "seat": seat, "outcome": outcome, "reason": rec.reason,
                                 "plies": rec.plies, "strikes": rec.strikes[seat],
                                 "log_bytes": os.path.getsize(log) if log and os.path.exists(log) else None})
    results["max_step"] = max(results["steps"], default=0.0)
    results["mean_step"] = (sum(results["steps"]) / len(results["steps"])) if results["steps"] else 0.0
    del results["steps"]                          # only max and mean go into the JSON
    report.dynamic.append(results)


def dynamic_phase(report: Report, games: int, rules: dict, in_process: bool = False) -> None:
    """Load the file in the sandbox and play `games` seat-swapped games against random_agent.

    Takes: report     - the Report from static_phase; its path is the file to load.
           games      - how many games. Game i plays seat 1 when i is even and seat 2 when odd,
                        and games 2k and 2k+1 share seed k, so they form a pair.
           rules      - {"allow_moves": ..., "win_length": ...}, the rule set to play under.
           in_process - True plays through InProcessRunner, for a platform where the sandbox
                        cannot start.
    Adds one dict of results to report.dynamic and two findings: the load (FAIL if the file did
    not load, WARN if __init__ comes close to its limit) and the games summary: FAIL on any
    forfeit, WARN on a strike, a note or a step close to the time limit, else PASS. The summary
    lists the first occurrence of each problem, with the line of the file where it was raised.
    """
    # Never `import agents` here: it would run the student's agents/student_agent.py outside the sandbox.
    from runners import InProcessRunner, SubprocessRunner
    saved = constants.current_rules()
    constants.configure_rules(**rules)                # in-process agents read the flags from constants
    try:
        with tempfile.TemporaryDirectory(prefix="validator_logs_") as log_dir:
            _play_games(report, games, rules,
                        lambda spec: InProcessRunner(spec) if in_process else SubprocessRunner(spec, log_dir=log_dir))
    finally:
        constants.configure_rules(**saved)            # a caller that validates more files keeps its own flags
    results = report.dynamic[-1]
    label = "load (in-process)" if in_process else "sandbox load"
    if results["load_error"]:
        error, line = _error_summary(results["load_error"], report.path)
        report.add("FAIL", f"{label}: {error}", line)
        return
    init = f"init_time {results['init_time']:.2f} s"
    slow_init = results["init_time"] > TIME_MARGIN * constants.INIT_TIME_LIMIT
    if slow_init:
        init += f", close to the {constants.INIT_TIME_LIMIT:g} s limit"
    if not in_process:
        init += ", rss " + ("unknown" if results["max_rss_mb"] is None else f"{results['max_rss_mb']} MB")
    report.add("WARN" if slow_init else "PASS", f"{label}: {init}")
    reasons = {}
    for g in results["games"]:
        if g["outcome"] == "L" and g["reason"] in FORFEIT_REASONS:
            reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    rules_text = f"allow_moves={rules['allow_moves']}, win_length={rules['win_length']}"
    summary = (f"{games} game{'s' * (games != 1)} vs random_agent ({rules_text}): "
               f"W{results['W']} D{results['D']} L{results['L']} | "
               f"max step {results['max_step']:.2f} s | strikes {sum(results['strikes'].values())}"
               f"{_counts_text(results['strikes'])} | forfeits {results['forfeits']}{_counts_text(reasons)}")
    if results["notes"]:
        summary += " | notes" + _counts_text(results["notes"])
    detail = list(results["examples"].values())
    slow_step = results["max_step"] > TIME_MARGIN * constants.STEP_TIME_LIMIT
    if slow_step and "soft_timeout" not in results["examples"]:
        detail.append(f"the slowest step() took {results['max_step']:.2f} s, close to the "
                      f"{constants.STEP_TIME_LIMIT:g} s limit")
    if results["forfeits"]:
        report.add("FAIL", summary, detail=detail)
    elif results["strikes"] or results["notes"] or slow_step:
        report.add("WARN", summary, detail=detail)
    else:
        report.add("PASS", summary)


def format_report(report: Report) -> str:
    """Return the report as the text main() prints.

    Takes:   report - a finished Report.
    Returns: a header naming the file and the interpreter, one line per finding (with its detail
             lines indented under it), and a last line "RESULT: PASS" or "RESULT: FAIL", with the
             warning count in parentheses when there is one, e.g. "RESULT: PASS (2 warnings)".
             After games that passed, a reminder of what a PASS means comes just before the last line.
    """
    lines = [f"VALIDATION of {report.path}  (python {platform.python_version()}, numpy {np.__version__})"]
    for f in report.findings:
        lines.append(f.text())
        lines += ["       " + d for d in f.detail]          # under the message, past "[FAIL] "
    if report.dynamic and report.verdict == "PASS":
        lines.append("Passing the validator does not mean your agent is good; it means the tournament can run it.")
    tail = f"RESULT: {report.verdict}"
    if report.warnings:
        tail += f" ({report.warnings} warning{'s' if report.warnings > 1 else ''})"
    lines.append(tail)
    return "\n".join(lines)


def main(argv=None) -> int:
    """Run the validator from the command line and return its exit code.

    Takes:   argv - a list of option strings, e.g. ["agents/student_agent.py", "--games", "2"];
                    None reads the real command line.
    Returns: 0 for PASS, 1 for FAIL, 2 when the validator itself raised. The report is printed
             and, with --json, also written to that file.
    """
    parser = argparse.ArgumentParser(description="Check a student_agent.py the way the tournament does: "
                                                 "first without running it, then in a few sandboxed games.")
    parser.add_argument("path", help="the file to check, e.g. agents/student_agent.py")
    parser.add_argument("--games", type=int, default=4,
                        help="games against random_agent, seats alternating (default 4)")
    parser.add_argument("--allow_moves", action=argparse.BooleanOptionalAction, default=constants.ALLOW_MOVES,
                        help=f"rule flag for the games (default from constants.py: {constants.ALLOW_MOVES})")
    parser.add_argument("--win_length", type=int, choices=[3, 4], default=constants.WIN_LENGTH,
                        help=f"rule flag for the games (default from constants.py: {constants.WIN_LENGTH})")
    parser.add_argument("--all_rules", action="store_true", default=False,
                        help="play the games under all four combinations of the rule flags")
    parser.add_argument("--static_only", action="store_true", default=False,
                        help="only check the file, without playing any game")
    parser.add_argument("--in_process", action="store_true", default=False,
                        help="play without the sandbox, where it cannot start (Windows)")
    parser.add_argument("--json", type=str, default=None, metavar="FILE", help="also write the report to FILE as JSON")
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error(f"--games must be at least 1, got {args.games}")   # exits with code 2, like any usage error
    try:
        report = static_phase(args.path)
        if report.verdict == "PASS" and not args.static_only:   # a file that failed statically may not even load
            combos = [{"allow_moves": a, "win_length": w} for a in (True, False) for w in (3, 4)] if args.all_rules \
                else [{"allow_moves": args.allow_moves, "win_length": args.win_length}]
            for rules in combos:
                dynamic_phase(report, args.games, rules, in_process=args.in_process)
        print(format_report(report))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        return 0 if report.verdict == "PASS" else 1
    except (Exception, SystemExit):
        # Exit code 2 keeps the validator's own error apart from a student FAIL. SystemExit too:
        # in-process, an agent whose autoplay is False can raise it past the referee.
        print("VALIDATOR ERROR\n" + traceback.format_exc())
        return 2


if __name__ == "__main__":
    sys.exit(main())
