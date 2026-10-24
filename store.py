"""The agent registry: the simulator finds each agent by the name it registered."""

AGENT_REGISTRY = {}     # registered name -> agent class


def register_agent(agent_name=""):
    """Return a class decorator that registers the class under `agent_name`.

    Example: @register_agent("student_agent") above a class makes it playable as
             `python simulator.py --player_1 student_agent`.
    Raises AssertionError if the name is already registered.
    """
    def decorator(func):
        if agent_name not in AGENT_REGISTRY:
            AGENT_REGISTRY[agent_name] = func
        else:
            raise AssertionError(
                f"Agent {AGENT_REGISTRY[agent_name]} is already registered."
            )
        return func
    return decorator
