"""The import rule for student_agent.py.

The validator checks a submission's import statements against it before running the file, and
the sandbox refuses every other import while the agent runs.
"""
from __future__ import annotations

import sys

# The standard library of the running interpreter, so the list always matches the Python that
# grades the submission.
ALLOWED_STDLIB = frozenset(sys.stdlib_module_names)
ALLOWED_OTHER = {"numpy", "helpers", "constants", "store", "agents", "agents.agent"}
ALLOWED_IMPORTS = ALLOWED_STDLIB | ALLOWED_OTHER


def is_allowed(module_name: str) -> bool:
    """Return True if a submission may import `module_name`.

    Takes: module_name - the dotted name as written in the import, e.g. "numpy.linalg".
    Allowed: every standard-library module, every part of numpy, helpers, constants and store.
    From the agents package only "agents" and "agents.agent" (the Agent base class): a built-in
    agent such as agents.random_agent is not available in the sandbox. Everything else is
    refused: the other files of this project, packages the tournament machine does not have
    (torch, scipy, matplotlib) and files of the student's own.
    Examples: "os" -> True, "numpy.linalg" -> True, "world" -> False, "agents.random_agent" -> False.
    """
    if module_name == "agents" or module_name.startswith("agents."):
        return module_name in ("agents", "agents.agent")
    return module_name.split(".")[0] in ALLOWED_IMPORTS
