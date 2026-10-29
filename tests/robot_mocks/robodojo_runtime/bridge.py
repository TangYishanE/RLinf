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

"""Small external VectorEnv boundary for CPU adapter contract tests."""

import numpy as np


class UnStableError(RuntimeError):
    pass


class VectorEnv:
    def __init__(self, task_config, n_envs, env_seeds):
        self.n_envs = n_envs
        self.states = np.tile(np.arange(14, dtype=np.float32), (n_envs, 1))
        self.resets = []
        self.reset_failures = 0
        self.closed = 0
        self.last_actions = None
        self.terminated = np.zeros(n_envs, bool)
        self.success = np.zeros(n_envs, bool)
        self.counts = None

    def reset(self, env_idx=None, env_seeds=None):
        if self.reset_failures:
            self.reset_failures -= 1
            raise UnStableError("layout moved")
        indices = (
            range(self.n_envs) if env_idx is None else np.atleast_1d(env_idx).tolist()
        )
        for index in indices:
            self.states[index] = np.arange(14) + env_seeds[index] / 1000000
        self.resets.append((list(indices), list(env_seeds)))

    def get_obs(self):
        return [
            {
                "full_image": np.full((8, 10, 3), i * 20, np.uint8),
                "state": self.states[i].copy(),
                "instruction": f"task {i}",
            }
            for i in range(self.n_envs)
        ]

    def step(self, actions):
        self.last_actions = actions.copy()
        self.states = actions[:, -1].copy()
        counts = (
            np.full(self.n_envs, actions.shape[1])
            if self.counts is None
            else self.counts
        )
        infos = [
            {"success": bool(self.success[i]), "executed_steps": int(counts[i])}
            for i in range(self.n_envs)
        ]
        return (
            self.get_obs(),
            self.success.astype(np.float32),
            self.terminated.copy(),
            np.zeros(self.n_envs, bool),
            infos,
        )

    def check_seeds(self, seeds):
        return [seed >= 0 for seed in seeds]

    def close(self, clear_cache=True):
        self.closed += 1
