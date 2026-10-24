"""The import rule for student_agent.py.

One rule, used in two places: the validator checks a submission's import statements before
running it, and the agent host refuses any other import while the agent runs.
"""
from __future__ import annotations

import sys

# Every module of the standard library, as the running interpreter lists it, so the list is
# always the one of the interpreter that grades the submission.
ALLOWED_STDLIB = frozenset(sys.stdlib_module_names)
ALLOWED_OTHER = {"numpy", "helpers", "constants", "store", "agents", "agents.agent"}
ALLOWED_IMPORTS = ALLOWED_STDLIB | ALLOWED_OTHER


def is_allowed(module_name: str) -> bool:
    """True if `import module_name` is permitted inside a submission.

    module_name: the dotted name as written in the import, e.g. "numpy.linalg".
    Every standard-library module and every part of numpy is allowed, and so are helpers,
    constants and store. Nothing else is: the other files of this project (world, runners,
    simulator), packages the tournament machine does not provide (torch, scipy, matplotlib)
    and files of the student's own. In the agents package only "agents" and "agents.agent"
    (the Agent base class) are allowed, so a submission cannot reuse a built-in agent such as
    agents.random_agent (that works in-process by accident and fails in the sandbox).
    Examples: "os" -> True, "numpy.linalg" -> True, "world" -> False, "agents.random_agent" -> False.
    """
    if module_name == "agents" or module_name.startswith("agents."):
        return module_name in ("agents", "agents.agent")
    return module_name.split(".")[0] in ALLOWED_IMPORTS
