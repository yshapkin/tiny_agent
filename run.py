from src.agent import TinyAgent
from src.llm import LLM


def main() -> None:
    llm = LLM(
        model="gemma4",
        think=False,
    )
    agent = TinyAgent(llm)
    print(agent.run("2 + 2 = ?"))


if __name__ == "__main__":
    main()
