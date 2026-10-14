"""End-to-end HTTP latency of POST /predict against a running ResNet-18 server.

Usage: python benches/http_bench.py PORT [MODE ...]

Modes: json1, json4 (JSON, 1 or 4 images); bin1, bin4 (raw little-endian float32 with
X-Batch-Size); append "o" (e.g. bin4o) to also request a binary response.

Request bodies are built before timing. Each sample covers sending the request, reading the
full response and decoding it, on one keep-alive connection. "io-overhead" is the end-to-end
median minus the server-reported inference time. Standard library only.
"""

import http.client
import json
import random
import statistics
import struct
import sys
import time

N = 150_528  # 3 x 224 x 224
RUNS = 30
WARMUP = 3


def make_image(rng):
    # Round-trip through float32 so JSON and binary carry the same values
    return [struct.unpack("<f", struct.pack("<f", rng.gauss(0, 1)))[0] for _ in range(N)]


def make_body(mode, images):
    if mode in ("json1", "json4"):
        # ~7 significant digits, as a float32-aware JSON encoder would write them
        rows = [[float(f"{v:.7g}") for v in im] for im in images[: int(mode[4])]]
        doc = {"input": rows[0]} if mode == "json1" else {"inputs": rows}
        return {"Content-Type": "application/json"}, json.dumps(doc).encode()
    if mode.startswith("bin"):
        batch = int(mode[3])
        headers = {"Content-Type": "application/octet-stream", "X-Batch-Size": str(batch)}
        if mode.endswith("o"):
            headers["Accept"] = "application/octet-stream"
        data = b"".join(struct.pack(f"<{N}f", *im) for im in images[:batch])
        return headers, data
    raise SystemExit(f"unknown mode: {mode}")


def main():
    port = int(sys.argv[1])
    modes = sys.argv[2:] or ["json1", "json4", "bin1", "bin4"]
    rng = random.Random(0)
    images = [make_image(rng) for _ in range(4)]
    conn = http.client.HTTPConnection("127.0.0.1", port)

    for mode in modes:
        headers, payload = make_body(mode, images)
        totals, inference = [], []
        for i in range(WARMUP + RUNS):
            start = time.perf_counter()
            conn.request("POST", "/predict", body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status != 200:
                raise SystemExit(f"{mode}: HTTP {response.status} {raw[:300]!r}")
            if "json" in response.getheader("Content-Type", ""):
                server_ms = json.loads(raw)["inference_time_ms"]
            else:
                server_ms = float(response.getheader("X-Inference-Time-Ms"))
                struct.unpack(f"<{len(raw) // 4}f", raw)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i >= WARMUP:
                totals.append(elapsed_ms)
                inference.append(server_ms)

        median = statistics.median(totals)
        p90 = statistics.quantiles(totals, n=10)[-1]
        server = statistics.median(inference)
        print(
            f"{mode:6} body={len(payload) / 1e6:5.2f}MB  e2e median={median:7.1f}ms "
            f"p90={p90:7.1f}ms  server inference={server:6.1f}ms  io-overhead={median - server:5.1f}ms"
        )


if __name__ == "__main__":
    main()
