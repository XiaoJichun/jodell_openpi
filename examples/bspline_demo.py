"""Run after installing packages/openpi-client and packages/openpi-async_bspline."""

import argparse
import time

import numpy as np
from openpi_async_bspline import AsyncBSplinePolicy
from openpi_client.base_policy import BasePolicy


class DemoPolicy(BasePolicy):
    def infer(self, obs):
        time.sleep(0.02)  # Simulated server latency.
        t = np.arange(obs["step"], obs["step"] + 50)
        return {"actions": np.column_stack((np.sin(t / 30), np.cos(t / 30)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="B-spline YAML file (use configs/demo.yaml for this 2D example)")
    args = parser.parse_args()
    with AsyncBSplinePolicy(DemoPolicy(), config=args.config) as policy:
        for step in range(120):
            action = policy.infer({"step": step})["actions"]
            if step % 20 == 0:
                print(f"{step=}: {action}")
            time.sleep(0.02)

    if policy.config.diagnostics_enabled:
        policy.save_diagnostics()
