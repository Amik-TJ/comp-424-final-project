"""The built-in agents. Importing this package registers all four of them.

Each module registers its own agent when it is imported, so the modules are imported here,
not the class names: renaming a class cannot break this import.
"""
from . import random_agent, greedy_agent, human_agent, student_agent  # noqa: F401
