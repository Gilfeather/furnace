"""FastAPI + ONNX Runtime with the same HTTP protocol as furnace, for `cmp-fastapi`.

This is the setup furnace competes with as a server: a Python web framework in front of
ONNX Runtime. It speaks furnace's binary protocol, so the Rust client is shared:

- GET /healthz, answering with this process's pid
- POST /predict: little-endian f32 input, batch size in `X-Batch-Size`; answers with
  little-endian f32 logits and the inference time in `X-Inference-Time-Ms`

Run with `python server.py --model ../../../models/resnet18.onnx --port 8000`.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Request, Response

CHANNELS, SIDE = 3, 224


def build_app(model: str, threads: int) -> FastAPI:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    app = FastAPI()

    # The pid lets the harness measure this process: on Windows a virtual environment's
    # python.exe is a launcher that runs the real interpreter as a child process
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "pid": os.getpid()}

    @app.post("/predict")
    async def predict(request: Request) -> Response:
        batch = int(request.headers.get("x-batch-size", "1"))
        body = await request.body()
        x = np.frombuffer(body, dtype="<f4").reshape(batch, CHANNELS, SIDE, SIDE)
        start = time.perf_counter()
        (logits,) = session.run(None, {input_name: x})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Response(
            content=np.ascontiguousarray(logits, dtype="<f4").tobytes(),
            media_type="application/octet-stream",
            headers={"X-Inference-Time-Ms": f"{elapsed_ms:.3f}"},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=0, help="0 = ONNX Runtime default")
    args = parser.parse_args()
    app = build_app(args.model, args.threads)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
