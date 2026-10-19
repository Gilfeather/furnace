"""Cold start: from launching the server to its first answers, as a scale-to-zero platform sees it.

    python benches/cold_start.py binary ./target/release/furnace --model-name resnet18
    python benches/cold_start.py docker ghcr.io/gilfeather/furnace:0.9.0

For each run it starts the server, polls GET /readyz every millisecond, then sends one ResNet-18
inference (binary float32, all zeros) and records:

- listening: the first HTTP answer of any kind (the process is accepting requests)
  (the port is probed with 5 ms connects, so a refused connection's delay is not counted)
- ready:     /readyz answered 200 (the warm-up inference has run)
- first inference: the first POST /predict answered 200, counted from launch

then stops the server. Reports the median and the range over the runs. Standard library only.
"""
import http.client
import socket
import statistics
import subprocess
import sys
import time

PORT = 3995
RUNS = 10
INPUT = bytes(3 * 224 * 224 * 4)  # one ResNet-18 image of zeros, float32


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
    c.request("GET", path)
    r = c.getresponse()
    r.read()
    return r.status


def predict():
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    c.request("POST", "/predict", INPUT, {"Content-Type": "application/octet-stream"})
    r = c.getresponse()
    r.read()
    return r.status


def port_open():
    """A connect with a short timeout: on Windows a refused connection takes about half a second
    to fail, which would otherwise be counted as start-up time"""
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=0.005).close()
        return True
    except OSError:
        return False


def launch(kind, target, args):
    if kind == "binary":
        return subprocess.Popen([target, "--port", str(PORT), *args],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.Popen(["docker", "run", "--rm", "--name", "furnace-cold-start",
                             "-p", f"{PORT}:3000", target, *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(kind, process):
    if kind == "docker":
        subprocess.run(["docker", "stop", "-t", "1", "furnace-cold-start"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.terminate()
    process.wait()


def run(kind, target, args):
    start = time.perf_counter()
    process = launch(kind, target, args)
    listening = ready = None
    deadline = start + 60
    while ready is None and time.perf_counter() < deadline:
        if not port_open():
            continue
        try:
            status = get("/readyz")
            listening = listening or time.perf_counter()
            if status == 200:
                ready = time.perf_counter()
        except (ConnectionError, OSError, http.client.HTTPException):
            pass
        time.sleep(0.001)
    assert ready, "never became ready"
    assert predict() == 200
    first = time.perf_counter()
    stop(kind, process)
    return [(t - start) * 1000 for t in (listening, ready, first)]


def main():
    kind, target, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    results = [run(kind, target, args) for _ in range(RUNS)]
    for i, name in enumerate(["listening", "ready", "first inference"]):
        values = [r[i] for r in results]
        print(f"{kind:6s} {name:16s} median {statistics.median(values):7.1f} ms"
              f"   (min {min(values):.1f}, max {max(values):.1f}, {RUNS} runs)")


if __name__ == "__main__":
    main()
