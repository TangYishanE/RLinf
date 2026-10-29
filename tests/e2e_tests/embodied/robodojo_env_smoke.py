# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the real RoboDojo reset/hold-action/close path on one GPU slot."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv


def run_smoke(cfg) -> dict:
    """Hold the measured joint state for two chunks and report interface shapes."""
    env = RoboDojoEnv(cfg, 1, 0, 1)
    try:
        obs, reset_info = env.reset(env_seeds=[0])
        initial = obs["main_images"].clone()
        actions = obs["states"][:, None].repeat(1, 8, 1)
        results = []
        for _ in range(2):
            obs, rewards, terminated, truncated, _ = env.chunk_step(actions)
            assert obs[0]["states"].shape == (1, 14)
            assert np.isfinite(obs[0]["states"].numpy()).all()
            assert rewards.shape == terminated.shape == truncated.shape == (1, 8)
            assert np.isfinite(rewards.numpy()).all()
            results.append(
                {
                    "terminated": terminated.any().item(),
                    "truncated": truncated.any().item(),
                }
            )
            if terminated.any() or truncated.any():
                break
        return {
            "reset": reset_info,
            "image_shape": list(initial.shape),
            "chunks": results,
        }
    finally:
        env.close()


def main() -> None:
    """Run the smoke test with the checked-in physical environment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(
        repo / "examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
    )
    cfg = OmegaConf.merge(
        cfg, {"runner": {"logger": {"log_path": "/tmp/robodojo-smoke"}}}
    )
    result = run_smoke(cfg)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
