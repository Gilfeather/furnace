"""A FastAPI server whose "model" returns its input, for comparing servers alone.

furnace-burn-bench serves the same echo model with furnace and with a hand-written axum
server, so the three differ only in the server around the model:

- GET /healthz, answering with this process's pid
- POST /infer: little-endian f32 values in, the same values out

Run with `python echo.py --port 8000 [--workers N]` (the requirements of
../../compare/fastapi-ort suffice).
"""

import argparse
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

app = FastAPI()


# On Windows a virtual environment's python.exe is a launcher that runs the real
# interpreter as a child process: the pid lets the harness measure the right one
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "pid": os.getpid()}


@app.post("/infer")
async def infer(request: Request) -> Response:
    x = np.frombuffer(await request.body(), dtype="<f4")
    return Response(
        content=np.ascontiguousarray(x, dtype="<f4").tobytes(),
        media_type="application/octet-stream",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(
        "echo:app",
        host="127.0.0.1",
        port=args.port,
        workers=args.workers,
        log_level="warning",
        app_dir=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == "__main__":
    main()
