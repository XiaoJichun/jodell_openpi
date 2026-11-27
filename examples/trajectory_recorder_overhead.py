"""CPU microbenchmark of recording overhead, not an end-to-end inference benchmark."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from openpi_trajectory_recorder import RecordedPolicy


class FixedPolicy:
    def __init__(self):
        self.result = {"actions": np.zeros(14)}

    def infer(self, obs):
        return self.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("Use at least 100 iterations")
    base = FixedPolicy()
    wrapped = RecordedPolicy(base, capacity=args.iterations, action_dim=14, hz=50)
    obs = {"state": np.zeros(14)}
    costs = {"baseline": [], "recorded": []}
    for repeat in range(8):
        # Alternate order to reduce systematic warmup/order effects.
        for name, policy in [("baseline", base), ("recorded", wrapped)][:: 1 if repeat % 2 else -1]:
            wrapped.recorder.clear()
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                policy.infer(obs)
            costs[name].append((time.perf_counter_ns() - start) / args.iterations / 1000)
    baseline, recorded = (float(np.median(costs[k][1:])) for k in ["baseline", "recorded"])
    result = {
        "iterations_per_batch": args.iterations,
        "batches": 8,
        "discard_first_batch": True,
        "baseline_us_per_call": baseline,
        "recorded_us_per_call": recorded,
        "added_us_per_call": recorded - baseline,
        "fraction_of_20ms_tick": (recorded - baseline) / 20000,
        "preallocated_bytes": wrapped.recorder.allocated_bytes,
        "scope": "CPU 14D NumPy vector microbenchmark. Not a GPU or hard-real-time guarantee.",
        "batch_us_per_call": costs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
