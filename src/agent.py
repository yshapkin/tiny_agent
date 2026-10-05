from .llm import LLM
from .memory import Memory
from .trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""
    def __init__(self, llm: LLM, memory: Memory):
        self.llm = llm
        self.memory = memory
        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)

        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step."""
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        return f"Executed action: {action}"