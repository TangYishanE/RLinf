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

"""World-model action, initialization, session and episode contracts on CPU."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

try:
    from opensora.action_stats import ActionNormalizer
except ModuleNotFoundError:
    pytest.skip("Requires the paired OpenSora checkout", allow_module_level=True)

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.world_model.backend.opensora import OpenSoraBackend
from rlinf.envs.sim.world_model.env import WorldModelEnv
from rlinf.envs.sim.world_model.registry import register_backend


class FrameVAE(torch.nn.Module):
    """External generation dependency with identity frame latents."""

    out_channels = 3

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def get_latent_size(self, size):
        return size


class ActionScheduler:
    """External sampler whose new pixels encode the supplied action chunk."""

    def sample(self, model, z, y, **kwargs):
        result = z.clone()
        result[:, :, -y.shape[1] :] = y[:, :, 0][:, None, :, None, None]
        return result


@register_backend("cpu_opensora_contract")
class CPUOpenSora(OpenSoraBackend):
    """Exercise production sessions with small external generation components."""

    def __init__(self, cfg, device):
        self.cfg, self.wm_cfg, self.device = cfg, cfg.world_model_cfg, device
        self.chunk = self.wm_cfg.chunk
        self.condition_frame_length = self.wm_cfg.condition_frame_length
        self.action_dim = self.wm_cfg.model.action_dim
        self.image_size = tuple(self.wm_cfg.image_size)
        self.num_frames = self.chunk + self.condition_frame_length
        self.z_mask_frame_num = self.chunk
        self.z_condition_frame_length = self.condition_frame_length
        self.inference_dtype = torch.float32
        self.is_vae_v1_2 = False
        self.vae, self.model, self.scheduler = (
            FrameVAE(),
            torch.nn.Identity(),
            ActionScheduler(),
        )
        self.action_stats = self._load_action_stats()
        self.model_args, self._sessions = {}, {}


def config(tmp_path, action_dim=14):
    import json

    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({"action": {"q01": [-1] * action_dim, "q99": [1] * action_dim}})
    )
    init = tmp_path / "init"
    init.mkdir(exist_ok=True)
    for i in range(2):
        np.save(
            init / f"episode{i}.npy",
            np.array(
                [
                    {
                        "image": np.full((4, 4, 3), 50 + i * 100, np.uint8),
                        "abs_action": np.zeros(action_dim, np.float32),
                        "instruction": f"task {i}",
                    }
                ],
                dtype=object,
            ),
        )
    return OmegaConf.create(
        {
            "backend": "cpu_opensora_contract",
            "wm_env_type": "robotwin",
            "action_dim": action_dim,
            "state_dim": 0,
            "is_eval": True,
            "reward_mode": "none",
            "action_key": "abs_action",
            "initial_image_path": str(init),
            "seed": 1,
            "video_cfg": {},
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "reward_coef": 1,
            "use_fixed_reset_state_ids": False,
            "group_size": 1,
            "max_episode_steps": 16,
            "world_model_cfg": {
                "chunk": 8,
                "condition_frame_length": 4,
                "image_size": [4, 4],
                "stats_path": str(stats),
                "model": {"action_dim": action_dim},
            },
        }
    )


@pytest.mark.parametrize("kind", ["numpy", "tensor", "bf16"])
def test_robotwin_action_dispatch_and_libero_regression(kind):
    actions = np.linspace(0, 1, 2 * 8 * 14, dtype=np.float32).reshape(2, 8, 14)
    value = (
        actions
        if kind == "numpy"
        else torch.from_numpy(actions).to(
            torch.bfloat16 if kind == "bf16" else torch.float32
        )
    )
    result = prepare_actions(
        value, "world_model", "openvla_oft", 8, 14, wm_env_type="robotwin"
    )
    expected = value.float().numpy() if isinstance(value, torch.Tensor) else value
    np.testing.assert_array_equal(result, expected)
    libero = np.zeros((1, 8, 7), np.float32)
    transformed = prepare_actions(
        libero, "world_model", "openvla_oft", 8, 7, wm_env_type="libero"
    )
    assert (transformed[..., -1] == 1).all()


@pytest.mark.parametrize(
    "actions", [np.zeros((1, 8, 7)), np.zeros((8, 14)), np.full((1, 8, 14), np.nan)]
)
def test_invalid_robotwin_actions(actions):
    with pytest.raises(ValueError):
        prepare_actions(
            actions, "world_model", "openvla_oft", 8, 14, wm_env_type="robotwin"
        )


def test_backend_session_lifecycle_and_shape_validation(tmp_path):
    cfg = config(tmp_path)
    backend = CPUOpenSora(cfg, torch.device("cpu"))
    initial = [[torch.zeros(3, 1, 4, 4) for _ in range(4)] for _ in range(2)]
    backend.open_session([0, 1], initial, torch.zeros(2, 4, 14), [0, 1])
    action = torch.full((2, 8, 14), 0.25, dtype=torch.bfloat16)
    output = backend.generate([0, 1], action)
    assert output.shape == (2, 3, 8, 4, 4)
    assert torch.all(output == 0.25)
    backend.generate([1], torch.zeros(1, 8, 14))
    backend.offload()
    backend.onload()
    with pytest.raises(ValueError, match="Expected actions"):
        backend.generate([0], torch.zeros(1, 25, 14))
    backend.close_session([0])
    backend.close_session([0])
    with pytest.raises(RuntimeError, match="without a session"):
        backend.generate([0], torch.zeros(1, 8, 14))
    assert backend.generate([1], torch.zeros(1, 8, 14)).shape[0] == 1


@pytest.mark.parametrize("benchmark", ["robotwin", "robodojo"])
def test_env_reset_chunks_partial_reset_and_timeout(tmp_path, benchmark):
    cfg = config(tmp_path)
    cfg.wm_env_type = benchmark
    env = WorldModelEnv(cfg, 2, 0, 1)
    obs, _ = env.reset(episode_indices=[0, 1])
    assert obs["states"].shape == (2, 0)
    assert obs["wrist_images"] is None
    assert obs["task_descriptions"] == ["task 0", "task 1"]
    actions = torch.full((2, 8, 14), 0.25)
    obs, rewards, done, truncated, _ = env.chunk_step(actions)
    assert not rewards.any() and not done.any() and not truncated.any()
    preserved = obs[0]["main_images"][1].clone()
    reset_obs, _ = env.reset(env_idx=[0], episode_indices=[1])
    torch.testing.assert_close(reset_obs["main_images"][1], preserved)
    assert env.elapsed_steps.tolist() == [0, 8]
    _, _, done, truncated, infos = env.chunk_step(actions)
    assert not done.any() and truncated[1, -1] and not truncated[0, -1]
    assert not infos[0]["episode"]["success_once"].any()
    assert env.current_obs.shape[3] == 12


@pytest.mark.parametrize("benchmark", ["robotwin", "robodojo"])
def test_training_without_reward_and_fake_state_rejected(tmp_path, benchmark):
    cfg = config(tmp_path)
    cfg.wm_env_type = benchmark
    cfg.is_eval = False
    with pytest.raises(ValueError, match="evaluation"):
        WorldModelEnv(cfg, 2, 0, 1)
    cfg.is_eval = True
    cfg.state_dim = 14
    with pytest.raises(ValueError, match="image-only"):
        WorldModelEnv(cfg, 2, 0, 1)


def test_libero_statistics_equivalence(tmp_path):
    cfg = config(tmp_path, 7)
    backend = CPUOpenSora(cfg, torch.device("cpu"))
    actions = np.random.default_rng(0).normal(size=(2, 8, 7)).astype(np.float32)
    np.testing.assert_allclose(backend._normalize_action(actions), actions, atol=2e-7)
    assert isinstance(backend.action_stats, ActionNormalizer)


def test_hydra_robotwin_config_is_inference_only():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    directory = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "embodiment"
        / "config"
        / "env"
    )
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(config_name="opensora_robotwin_place_empty_cup")
    assert cfg.backend == "opensora" and cfg.wm_env_type == "robotwin"
    assert cfg.action_dim == cfg.world_model_cfg.model.action_dim == 14
    assert cfg.world_model_cfg.chunk == 8
    assert (
        cfg.world_model_cfg.model.pad_action_num
        == cfg.world_model_cfg.condition_frame_length
        == 4
    )
    assert cfg.state_dim == 0 and cfg.reward_mode == "none" and cfg.is_eval


def test_reset_dimension_mismatch_is_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg.action_dim = 7
    cfg.wm_env_type = "libero"
    env = WorldModelEnv(cfg, 2, 0, 1)
    with pytest.raises(ValueError, match="reset action dimension"):
        env.reset(episode_indices=[0, 1])


@pytest.mark.parametrize("benchmark", ["robotwin", "robodojo"])
def test_autoregressive_episode_evaluation(tmp_path, benchmark):
    from opensora.robotwin_evaluation import rollout_episode

    cfg = config(tmp_path)
    cfg.wm_env_type = benchmark
    backend = CPUOpenSora(cfg, torch.device("cpu"))
    rgb = np.full((18, 4, 4, 3), 128, np.uint8)
    predictions, metrics = rollout_episode(backend, rgb, np.zeros((17, 14), np.float32))
    assert predictions.shape == (16, 4, 4, 3)
    assert metrics["evaluated_steps"] == 16 and metrics["discarded_tail_steps"] == 1
    assert len(metrics["mse_by_horizon"]) == 16
    assert np.isfinite(metrics["psnr_by_horizon"]).all()
    with pytest.raises(RuntimeError, match="without a session"):
        backend.generate([0], torch.zeros(1, 8, 14))


def test_state_targets_require_explicit_offline_opt_in(tmp_path):
    import json
    from pathlib import Path

    cfg = config(tmp_path)
    path = Path(cfg.world_model_cfg.stats_path)
    stats = json.loads(path.read_text())
    stats["action_source"] = "state_target"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="recorded state targets"):
        CPUOpenSora(cfg, torch.device("cpu"))
    cfg.world_model_cfg.allow_state_target_actions = True
    backend = CPUOpenSora(cfg, torch.device("cpu"))
    assert backend.action_stats.action_source == "state_target"


def test_backend_rejects_invalid_geometry_before_loading_weights(tmp_path):
    cfg = config(tmp_path)
    cfg.world_model_cfg.vae = {"type": "SDXL"}
    cfg.world_model_cfg.model.pad_action_num = 3
    with pytest.raises(ValueError, match="pad_action_num"):
        OpenSoraBackend(cfg, torch.device("cpu"))
    cfg.world_model_cfg.model.pad_action_num = 4
    cfg.world_model_cfg.chunk = 0
    with pytest.raises(ValueError, match="positive"):
        OpenSoraBackend(cfg, torch.device("cpu"))


def test_backend_rejects_cross_benchmark_statistics(tmp_path):
    import json
    from pathlib import Path

    cfg = config(tmp_path)
    cfg.wm_env_type = "robodojo"
    path = Path(cfg.world_model_cfg.stats_path)
    stats = json.loads(path.read_text())
    stats["benchmark"] = "robotwin"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="benchmark"):
        CPUOpenSora(cfg, torch.device("cpu"))
    stats["benchmark"] = "robodojo"
    path.write_text(json.dumps(stats))
    assert CPUOpenSora(cfg, torch.device("cpu")).action_stats.benchmark == "robodojo"
