import time

import numpy as np

ROUNDS = 3
WARMUP = 20
MEASURE = 100


def run_protocol(sessions, x):
    feed = {"input": np.ascontiguousarray(x, dtype=np.float32)}
    raw = {name: [] for name in sessions}
    for _ in range(ROUNDS):
        for name, sess in sessions.items():
            for _ in range(WARMUP):
                sess.run(None, feed)
            times = []
            for _ in range(MEASURE):
                t0 = time.perf_counter()
                sess.run(None, feed)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e3)
            raw[name].append(times)
    res = {}
    for name, rounds in raw.items():
        all_ms = np.concatenate(rounds)
        res[name] = {
            "p50_ms": float(np.percentile(all_ms, 50)),
            "p95_ms": float(np.percentile(all_ms, 95)),
            "p50_by_round_ms": [float(np.percentile(r, 50)) for r in rounds],
            "raw_ms_by_round": rounds,
        }
    return res


def intervals_disjoint(a, b):
    return max(a) < min(b) or max(b) < min(a)
