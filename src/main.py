from .agent import TinyAgent
from .llm import LLM
from .memory import Memory


def main() -> None:
    llm = LLM(
        model="ggml-org/gemma-4-E2B-it-GGUF:Q8_0",
        base_url="http://localhost:9931/v1",
        temperature=0.8,
        think=True
    )
    memory = Memory()
    agent = TinyAgent(llm=llm, memory=memory)

    conversation = [
        "Hi! We are Alice and Bob, authors of 'A Beginner's Guide to Tiny Agents'.",
        "Hi! What are our names?",
    ]

    for turn, prompt in enumerate(conversation, start=1):
        print(f"[Turn {turn}] User:  {prompt}")
        print(f"[Turn {turn}] Agent: {agent.run(prompt)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
