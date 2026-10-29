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

"""RoboDojo vector environment contracts against the external runtime boundary."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv, state_action_permutations


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "robot_mocks")
    )
    return OmegaConf.create(
        {
            "seed": 0,
            "group_size": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "use_custom_reward": True,
            "use_fixed_reset_state_ids": False,
            "reward_coef": 1.0,
            "state_action_order": "aloha",
            "max_episode_steps": 8,
            "is_eval": False,
            "enable_offload": False,
            "task_config": {
                "task_name": "put_bottles_into_dustbin",
                "action_dims": [6, 6, 1, 1],
            },
        }
    )


def test_registration_is_lazy():
    assert get_env_cls("robodojo") is RoboDojoEnv


def test_state_and_action_order_through_public_env(cfg):
    env = RoboDojoEnv(cfg, 2, 0, 1)
    obs, _ = env.reset(env_seeds=[0, 1])
    np.testing.assert_allclose(
        obs["states"][0], [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    )
    actions = torch.arange(14).float().reshape(1, 1, 14).repeat(2, 4, 1).bfloat16()
    obs, reward, terminated, truncated, _ = env.step(actions)
    np.testing.assert_array_equal(
        env.venv.last_actions[0, 0], [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    )
    np.testing.assert_array_equal(obs["states"][0], np.arange(14))
    assert not reward.any() and not terminated.any() and not truncated.any()
    assert env.elapsed_steps.tolist() == [4, 4]
    env.close()
    env.close()
    assert env.venv.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.step(actions)


def test_partial_reset_and_early_termination_mask(cfg):
    cfg.auto_reset = True
    env = RoboDojoEnv(cfg, 2, 0, 1)
    env.reset(env_seeds=[0, 1])
    env.venv.terminated[0] = True
    env.venv.success[0] = True
    env.venv.counts = np.array([2, 4])
    obs, reward, terminated, truncated, infos = env.chunk_step(
        np.ones((2, 4, 14), np.float32)
    )
    assert reward[0, -1] == 1 and terminated[0, -1] and not truncated.any()
    assert infos[0]["valid_action_mask"].tolist() == [
        [True, True, False, False],
        [True] * 4,
    ]
    assert infos[0]["final_info"]["episode"]["success_once"].tolist() == [True, False]
    assert env.elapsed_steps.tolist() == [0, 4]
    assert env.venv.resets[-1][0] == [0]
    assert obs[0]["states"].shape == (2, 14)


def test_failed_termination_is_not_success(cfg):
    env = RoboDojoEnv(cfg, 1, 0, 1)
    env.reset()
    env.venv.terminated[0] = True
    _, reward, terminated, _, info = env.step(np.zeros((1, 4, 14)))
    assert terminated[0] and reward[0] == 0
    assert not info["episode"]["success_once"].any()


def test_timeout_and_ignored_success(cfg):
    cfg.ignore_terminations = True
    env = RoboDojoEnv(cfg, 1, 0, 1)
    env.reset()
    env.venv.terminated[0] = True
    env.venv.success[0] = True
    _, _, terminated, truncated, info = env.step(np.zeros((1, 8, 14)))
    assert not terminated.any() and truncated[0]
    assert info["episode"]["success_at_end"][0]


@pytest.mark.parametrize(
    "actions", [np.zeros((1, 4, 7)), np.full((1, 4, 14), np.nan), np.zeros((1, 0, 14))]
)
def test_invalid_actions(cfg, actions):
    env = RoboDojoEnv(cfg, 1, 0, 1)
    with pytest.raises(ValueError):
        env.step(actions)


def test_reset_retries_are_reported(cfg):
    env = RoboDojoEnv(cfg, 1, 0, 1)
    env.venv.reset_failures = 2
    _, info = env.reset(env_seeds=[1])
    assert info["reset_attempts"] == 3 and info["reset_seeds"] == [1]
    with pytest.raises(ValueError, match="every global slot"):
        env.reset(env_seeds=[])


def test_grouped_seeds_and_terminal_offload(cfg):
    cfg.group_size = 2
    env = RoboDojoEnv(cfg, 4, 0, 1)
    assert env.reset_state_ids[0] == env.reset_state_ids[1]
    assert env.reset_state_ids[2] == env.reset_state_ids[3]
    cfg.enable_offload = True
    with pytest.raises(ValueError, match="terminal"):
        RoboDojoEnv(cfg, 4, 0, 1)


@pytest.mark.parametrize(
    "order,dims", [("bad", [6, 6, 1, 1]), ("aloha", [6, 0, 1, 1]), ("native", [6])]
)
def test_invalid_action_layout(order, dims):
    with pytest.raises(ValueError):
        state_action_permutations(order, dims)


def test_world_model_dispatch_and_hydra_config(monkeypatch):
    actions = np.zeros((1, 8, 14), np.float32)
    np.testing.assert_array_equal(
        prepare_actions(
            actions, "world_model", "openvla_oft", 8, 14, wm_env_type="robodojo"
        ),
        actions,
    )
    directory = Path(__file__).resolve().parents[2] / "examples/embodiment/config/env"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(config_name="opensora_robodojo_put_bottles_into_dustbin")
    assert config.wm_env_type == "robodojo" and config.action_dim == 14
    assert config.reward_mode == "none" and config.state_dim == 0
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "evaluations/robodojo")
    ):
        config = compose(config_name="robodojo_put_bottles_into_dustbin_pi05_eval")
    assert config.env.eval.env_type == "robodojo"
    assert config.env.eval.state_action_order == "aloha"
    assert config.rollout.model.openpi.torch_compile is False
    assert Path(config.env.eval.seeds_path).is_file()


def test_install_script_preflight_and_torch_default(tmp_path):
    script = Path(__file__).resolve().parents[2] / "requirements/install.sh"
    source = script.read_text().rsplit('main "$@"', 1)[0]
    path = tmp_path / "installer.sh"
    path.write_text(source)
    result = subprocess.run(
        ["bash", "-c", f'source "{path}"; PLATFORM=amd; install_robodojo_env'],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "requires" in result.stderr
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{path}"; ENV_NAME=robodojo; TORCH_VERSION=""; apply_env_default_torch; test "$TORCH_VERSION" = 2.7.0',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gpu_smoke_driver_contract_with_vendor_fake(cfg):
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1] / "e2e_tests/embodied/robodojo_env_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("robodojo_env_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg.max_episode_steps = 32
    result = module.run_smoke(cfg)
    assert result["reset"]["reset_seeds"] == [0]
    assert len(result["chunks"]) == 2
    assert result["image_shape"] == [1, 8, 10, 3]


def test_eval_script_benchmark_inference(tmp_path):
    script = Path(__file__).resolve().parents[2] / "evaluations/run_eval.sh"
    source = script.read_text().split("infer_benchmark()", 1)[1]
    source = "infer_benchmark()" + source.split("\n}", 1)[0] + "\n}\n"
    path = tmp_path / "eval.sh"
    path.write_text(source)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{path}"; infer_benchmark robodojo_put_bottles_into_dustbin_pi05_eval',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and result.stdout.strip() == "robodojo"
