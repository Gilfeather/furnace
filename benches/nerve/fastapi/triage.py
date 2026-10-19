"""A FastAPI server doing the nerve demo's triage in a handler, for comparing servers alone.

furnace-nerve-bench serves the same work with furnace's nervous system and with a hand-written
axum server:

- GET /healthz, answering with this process's pid
- POST /events: a text line in. A 64-bit sketch of the body is its feature; a bounded
  habituation table (1024 entries, exponential decay with a 30 s time constant, gain 0.5)
  attenuates repeats. Below the 0.6 threshold: 204 (quiet). A line containing ALERT: 200
  (the alarm acted). Anything else: 202 (to the brain, counted).

Run with `python triage.py --port 8000` (FastAPI and uvicorn suffice).
"""

import argparse
import math
import os
import time
from collections import OrderedDict

import uvicorn
from fastapi import FastAPI, Request, Response

TAU = 30.0
THRESHOLD = 0.6
GAIN = 0.5
CAPACITY = 1024

app = FastAPI()
table: "OrderedDict[int, tuple[float, float]]" = OrderedDict()
brain = 0


def attenuation(sketch: int, now: float) -> float:
    entry = table.get(sketch)
    if entry is None:
        if len(table) >= CAPACITY:
            table.popitem(last=False)
        table[sketch] = (GAIN, now)
        return 0.0
    a, seen = entry
    decayed = a * math.exp(-(now - seen) / TAU)
    table[sketch] = (decayed + (1.0 - decayed) * GAIN, now)
    table.move_to_end(sketch)
    return decayed


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "pid": os.getpid()}


@app.post("/events")
async def events(request: Request) -> Response:
    global brain
    body = await request.body()
    # Python's own 64-bit hash of the bytes stands in for furnace's sketch
    if 1.0 - attenuation(hash(body), time.monotonic()) < THRESHOLD:
        return Response(status_code=204)
    if b"ALERT" in body:
        return Response(content=b"alarm", status_code=200, media_type="text/plain")
    brain += 1
    return Response(status_code=202)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(
        "triage:app",
        host="127.0.0.1",
        port=args.port,
        workers=1,
        log_level="warning",
        app_dir=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == "__main__":
    main()
