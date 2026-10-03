from .llm import LLM
from .trajectory import Trajectory


class TinyAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        messages = [{"role": "user", "content": task}]
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str:
        return f"Executed action: {action}"