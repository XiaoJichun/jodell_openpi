"""Replay existing real traces to verify the recording plugin; does not run a model."""

import argparse
from pathlib import Path

import numpy as np
from openpi_trajectory_recorder import RecordedPolicy
from openpi_trajectory_recorder.analysis import analyze


class ReplayPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def infer(self, obs):
        result = {"actions": self.actions[self.index]}
        self.index += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/aloha_sim/a100_results_20261125"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recordings = []
    for mode in ("sync", "rtc", "bspline"):
        with np.load(args.input / mode / "seed_0.npz") as trace:
            actions, states = trace["actions"], trace["states"]
        policy = RecordedPolicy(ReplayPolicy(actions), capacity=len(actions), action_dim=14, hz=50)
        for state in states:
            policy.infer({"state": state})
        np.testing.assert_array_equal(policy.recorder.actions, actions)
        np.testing.assert_array_equal(policy.recorder.states, states)
        recordings.append((mode, policy))
    # All inference/replay calls are finished before any save or plotting.
    paths = [
        policy.save(args.output / mode / "seed_0.npz", metadata={"mode": mode, "replay": True})
        for mode, policy in recordings
    ]
    analyze(paths, args.output / "comparison", labels=[mode for mode, _ in recordings])
    print(f"Replay verified; offline figures: {args.output / 'comparison'}")


if __name__ == "__main__":
    main()
