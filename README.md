# TinyAgent

TinyAgent is a minimal Python agent that sends a prompt to an
OpenAI-compatible chat completions API and records the response in an
in-memory trajectory.

## Model hosting

TinyAgent talks to any OpenAI-compatible chat completions API. For local
model hosting, you can use [Llama · Your AI, on your computer](https://llama.app/).

## Running

From the repository root, run the agent as a module:

```bash
python -m src.main
```