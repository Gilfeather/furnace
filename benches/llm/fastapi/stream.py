"""A FastAPI server streaming mock OpenAI chat chunks, built like vLLM's API server.

furnace-llm-bench streams the same answer from furnace and from a hand-written axum server, so
the servers differ only in the server layer around a mock engine:

- GET /healthz, answering with this process's pid
- POST /v1/chat/completions: the request is parsed and ignored; the answer is a role chunk,
  TOKENS chunks of one token ("x"), token i at i * TOKEN_MS after the request arrived, the
  finish chunk and [DONE]

Run with `python stream.py --port 8000 --tokens 64 --token-ms 20` (the requirements of
../../compare/fastapi-ort suffice). One uvicorn worker: an LLM server's engine lives in one
process.
"""

import argparse
import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
TOKENS = 64
TOKEN_SECONDS = 0.02


# On Windows a virtual environment's python.exe is a launcher that runs the real
# interpreter as a child process: the pid lets the harness measure the right one
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "pid": os.getpid()}


def chunk(delta: dict, finish_reason) -> str:
    body = {
        "id": "chatcmpl-fastapi",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body, separators=(',', ':'))}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    await request.json()
    start = time.perf_counter()

    async def events():
        yield chunk({"role": "assistant", "content": ""}, None)
        for i in range(TOKENS):
            delay = start + i * TOKEN_SECONDS - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            yield chunk({"content": "x"}, None)
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def fine_timers() -> None:
    """Timers to the millisecond, as furnace-llm-bench's Rust servers ask for: Windows wakes
    sleeping threads every 15.6 ms by default"""
    if os.name == "nt":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def main() -> None:
    global TOKENS, TOKEN_SECONDS
    fine_timers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--token-ms", type=float, default=TOKEN_SECONDS * 1e3)
    args = parser.parse_args()
    TOKENS = args.tokens
    TOKEN_SECONDS = args.token_ms / 1e3
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
