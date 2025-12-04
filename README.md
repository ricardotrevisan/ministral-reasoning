# model-eval-ministral-3-8B-reasoning

Local evaluation and chat experiments with the `Ministral-3-8B-Reasoning` model using `llama.cpp` and the OpenAI-compatible `/v1/chat/completions` API exposed by `llama-server`.

## Overview

This repo is a small playground to:
- Run `Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf` via `llama.cpp` with CUDA.
- Hit the local HTTP server with simple curl calls.
- Run an interactive streaming chat client in Python (`streaming.py`).

It assumes you already have `llama.cpp` built on your machine (optionally with CUDA), and that the `.gguf` model file is located in this directory.

## Prerequisites

- Python 3.9+ (or similar)
- A working build of `llama.cpp` with CUDA support (optional but recommended)
- `Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf` in the project root

Python dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

## Running llama.cpp server

From within this repo, start the `llama.cpp` HTTP server (adjust paths/flags as needed):

```bash
../llama.cpp/build/bin/llama-server \
  -m Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --threads 8 \
  --gpu-layers 999
```

Once the server is up, you can test it with curl.

### Legacy completions endpoint

```bash
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "<s>[INST] Explique 2+2. [/INST]",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 200,
        "stop": ["</s>"]
      }'
```

### OpenAI-style chat completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf",
    "messages": [
      {"role": "user", "content": "Explique 2+2 de forma objetiva e curta."}
    ]
  }'
```

With an explicit system message:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf",
    "messages": [
      {
        "role": "system",
        "content": "Você é um assistente conciso. Responda de forma curta e direta."
      },
      {
        "role": "user",
        "content": "Explique 2+2 em uma frase."
      }
    ],
    "max_tokens": 32,
    "temperature": 0.2
  }'
```

## Interactive streaming client (`streaming.py`)

The `streaming.py` script is a simple terminal chat client that:
- Maintains the conversation history in a `messages` list.
- Sends requests to `http://127.0.0.1:8080/v1/chat/completions`.
- Streams tokens as they arrive, while also buffering the full assistant reply.

To run it (with the server already running):

```bash
python streaming.py
```

You can type messages in Portuguese or any other language. Type `sair`, `exit`, `quit` or `bye` to end the session.

## Notes

Additional build and usage notes for `llama.cpp` and CUDA live in `.notes.md`. That file is mainly for personal setup reminders; `README.md` is the high-level overview and quickstart for the project.

