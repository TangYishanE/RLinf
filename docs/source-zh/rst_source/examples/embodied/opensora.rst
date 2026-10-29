基于 OpenSora 世界模型的强化学习
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/hpcaitech/Open-Sora-Demo/main/readme/icon.png
   :align: center
   :width: 30%

   作为动作条件世界模型的 OpenSora。

使用 **动作条件 OpenSora 世界模型** 作为环境后端，**无需真实机器人或物理仿真器** 即可闭环训练
VLA 策略。OpenSora 根据当前观测与动作序列生成未来视频帧，因此可以在“想象”的 rollout 上用
强化学习（GRPO/PPO）优化策略。

概览
----------------------------------------

在 OpenSora 世界模型模拟的 LIBERO 套件上用 GRPO 训练 OpenVLA-OFT。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Spatial · Object

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装 → 下载 VLA 模型 → 下载 OpenSora 世界模型权重与初始化数据 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 OpenVLA-OFT SFT checkpoint · OpenSora 世界模型权重与初始化数据集（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为世界模型，OpenSora 原则上可以通过一致接口适配任意任务。RLinf 目前提供两个 LIBERO 套件的权重和初始化数据：

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - OpenSora
     - LIBERO-Spatial
     - ``RLinf/RLinf-OpenSora-LIBERO-Spatial``
     - 使用 OpenSora 作为 LIBERO spatial 任务的学习型仿真器。
   * - OpenSora
     - LIBERO-Object
     - ``RLinf/RLinf-OpenSora-LIBERO-Object``
     - 在视频世界模型中 rollout 物体操作动力学。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 由初始化帧启动、世界模型生成的 RGB 帧，形状为 ``[B, 256, 256, 3]``。
   * - Action
     - 归一化并 tokenized 后用于条件生成的 7 维连续动作。
   * - Reward
     - 世界模型 reward classifier 输出，范围为 ``[0, 1]``。
   * - Prompt
     - 用于条件化视频世界模型的自然语言任务描述。

与传统仿真器不同，OpenSora 没有 ``reset()``：它需要初始化帧和任务描述，因此需要下载初始化数据集并在配置中指向它。

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.4-opensora``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-opensora
      # 国内镜像加速：infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-opensora

   # 进入容器后，切换到 OpenVLA-OFT 虚拟环境：
   source switch_env openvla-oft

**选项 2：自定义环境** —— 安装套件 ``--env opensora``：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --model openvla-oft --env opensora
   source .venv/bin/activate

VLA 模型下载
----------------------------------------

下载 OpenVLA-OFT SFT checkpoint：

.. code:: bash

   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-object-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-traj1

   # 方法 2：使用 huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir Openvla-oft-SFT-libero-spatial-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero-object-traj1 --local-dir Openvla-oft-SFT-libero-object-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir Openvla-oft-SFT-libero-goal-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero10-traj1 --local-dir Openvla-oft-SFT-libero10-traj1

下载完成后，在配置中设置模型路径与 ``unnorm_key``：

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
         unnorm_key: libero_90_no_noops_trajall # 对于 RLinf-OpenVLAOFT-LIBERO-130-Base-Lora 模型，使用 libero_130_no_noops_trajall

世界模型下载
----------------------------------------

除 VLA 模型外，还需下载 OpenSora 权重与用于仿真初始化的数据集。当前 RLinf 仅提供
libero-spatial 与 libero-object 的权重与数据（各 suite 的 OpenSora 权重均基于 VLA 模型
rollout 的 3000 条轨迹构建）：

.. code:: bash

   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenSora-LIBERO-Spatial
   git clone https://huggingface.co/RLinf/RLinf-OpenSora-LIBERO-Object

   # 方法 2：使用 huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenSora-LIBERO-Spatial --local-dir RLinf-OpenSora-LIBERO-Spatial
   hf download RLinf/RLinf-OpenSora-LIBERO-Object --local-dir RLinf-OpenSora-LIBERO-Object

``RLinf-OpenSora-LIBERO-Spatial`` 的目录结构如下：

.. code-block:: text

    RLinf-OpenSora-LIBERO-Spatial/
        ├── dataset_statistics.json             # 数据集归一化统计信息
        ├── dataset/                            # 仿真初始化数据集
        │   ├── traj0.npy
        │   ├── traj1.npy
        │   ├── ...
        │   └── trajN.npy
        ├── model-00001.safetensors              # 世界模型权重文件
        ├── model.safetensors.index.json
        ├── config.json
        ├── resnet_rm.pth                        # 奖励模型权重文件
        └── vae/                                 # VAE 模型权重文件

下载完成后，在配置中设置世界模型路径：

.. code:: yaml

    env:
        train:
            opensora_wm_hf_ckpt_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/

运行
----------------------------------------

**1. 模型参数**

以 OpenVLA-OFT 为例，在 ``actor.model`` 中配置：

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/model/Openvla-oft-SFT-libero-spatial-traj1/"    # SFT 模型路径
       model_type: "openvla_oft"                                             # 模型类型设置为 openvla_oft
       use_proprio: False                                                    # 是否使用本体感觉信息
       num_images_in_input: 1                                                # 输入图像数量
       num_action_chunks: 8                                                  # 动作块数量
       unnorm_key: "libero_spatial_no_noops"                                 # 动作归一化键（与 SFT 一致）

由于世界模型不提供本体信息、不生成腕部视角且 chunk 固定，``use_proprio`` 默认为 ``False``，
``num_images_in_input`` 默认为 ``1``，``num_action_chunks`` 默认为 ``8``。

**2. 环境配置**

.. code-block:: yaml

   # 推荐训练使用 opensora_libero_spatial，评估使用 libero_spatial
   env/train: opensora_libero_spatial
   env/eval: libero_spatial
   env:
      train:
         opensora_wm_hf_ckpt_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/

   # 在 env/train/opensora_libero_spatial.yaml 中：
   env_type: world_model
   backend: opensora
   wm_env_type: libero
   # world model 初始化的初始图像路径
   initial_image_path: ${env.train.opensora_wm_hf_ckpt_path}/dataset_for_rlinf_world_model_init/base_policy_rollout_buffer
   # 不建议修改 world_model_cfg 中的参数
   world_model_cfg:
      stats_path: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/best_wm_ckpt/base_policy/dataset_statistics.json
      chunk: 8                     # 与训练和 VLA 推理长度对齐，默认 8
      condition_frame_length: 4    # 上下文记忆长度，默认 4
      model:
         from_pretrained: /Pathto/model/RLinf-OpenSora-LIBERO-Spatial/best_wm_ckpt/base_policy/model

**3. 启动**

OpenVLA-OFT + GRPO 使用 ``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml``：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft

可视化与结果
----------------------------------------

关注未归一化的回合成功率指标 ``env/success_once``。各项指标的含义见
:doc:`训练指标 <../../reference/metrics>`。可通过以下配置保存生成的 rollout 视频：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

我们评估所有 ``task_id`` × ``trial_id`` 组合——每个套件 500 个环境（10 个任务 × 50 个试次）。
SFT（LoRA-base）模型设置 ``do_sample = False``；RL 训练模型在 ``rollout.sampling_params`` 中设置
``do_sample = True``、``temperature_train = 1.6``，并启用 ``env.train.rollout_epoch=2``。

.. note::

    选择 OpenSora 作为世界模型模拟器的动机来源于 `WMPO <https://arxiv.org/abs/2511.09515>`_；
    在训练世界模型时我们也参考了 `OpenSora <https://github.com/RLinf/opensora>`_。

.. list-table:: **使用 OpenSora 模拟器的 LIBERO 任务组评测结果**
    :header-rows: 1
    :widths: 50 25 25

    * - 模型
      - Spatial
      - Object
    * - OpenVLA-OFT (LoRA-base)
      - 61.2%
      - 36.7%
    * - OpenVLA-OFT（OpenSora 作为世界模型的 RLinf-GRPO）
      - 75.5%
      - 64.5%
    * - **效果提升**
      - **+14.3%**
      - **+27.8%**

RoboTwin 推理接口
----------------------------------------

使用 ``env/opensora_robotwin_place_empty_cup.yaml`` 配置，将按 ALOHA 顺序排列的 14 维关节目标交给目标域 OpenSora checkpoint。将 ``initial_image_path``、``world_model_cfg.stats_path`` 和模型/VAE 路径指向本地资产。初始化数据使用 ``action_key: abs_action``。

该配置设置 ``is_eval: True``、``reward_mode: none`` 和 ``state_dim: 0``，返回头部相机图像、空本体状态和空腕部图像，奖励与预测成功率均为零。选择按相同输入训练的 image-only policy，并将实际执行 horizon 与世界模型的八步动作块对齐。policy RL 还需要训练好的目标奖励模型和真实执行命令数据。

在训练统计中保留动作来源。``action_source: state_target`` 表示以记录的关节状态作为示范目标，不代表已验证与仿真命令等价。后端默认拒绝这种统计，只有在相同表示的离线评测中显式设置 ``world_model_cfg.allow_state_target_actions: True`` 才可使用。训练与推理共用 OpenSora 检出中的 ``ActionNormalizer``，因此应搭配使用两个更新后的仓库。


RoboDojo 数据与仿真接口
----------------------------------------

使用 ``env/opensora_robodojo_put_bottles_into_dustbin.yaml``，将 ALOHA 顺序的 14 维关节目标交给 image-only 世界模型。启动前设置初始图像、RoboDojo 训练统计、模型 checkpoint 与 VAE 路径。policy 应使用单路头部图像、空本体状态和八步动作。该配置返回零奖励；物理任务成功率应在 RoboDojo 仿真器中评测。

在配套 OpenSora 检出中执行 ``python -m scripts.convert_robodojo --manifest source.json --output converted``，转换显式选择的 HDF5 字段。在源清单声明四组关节路径、native/ALOHA 顺序、解码后的 RGB/BGR 顺序、控制频率、数据划分和时间对齐。转换器生成 N+1 帧 RGB 与 N 个动作，只使用 train 回合拟合归一化。运行 ``python -m scripts.train_robodojo --config config/train_robodojo_put_bottles_into_dustbin.yaml`` 训练，通过 ``python -m scripts.evaluate_robodojo`` 评测留出轨迹。各命令的必填路径见 ``--help``。微调应使用对应任务数据；公开的两条 ARX-X5 demo 是堆碗任务。

若使用记录的关节状态，设置 ``action_source: state_target``、``alignment: incoming``，以 state t+1 作为 image t 到 image t+1 的目标。只在相同表示的离线评测中启用 ``allow_state_target_actions``。仿真命令评测需要记录的真实命令，以及按同一命令接口训练的 checkpoint。

在 Linux x86_64 与 NVIDIA CUDA 环境安装独立的物理仿真器：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/data/robodojo
   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   python tests/e2e_tests/embodied/robodojo_env_smoke.py --output robodojo-smoke.json

此命令安装固定版本的 runtime、Isaac Sim 5.1 与 IsaacLab 依赖，然后重置布局 0，保持测得关节目标并执行两次动作块。先将仿真资产放到 ``$ROBODOJO_ASSETS_ROOT/Assets``。安装器为 Isaac Sim 设置 ``OMNI_KIT_ACCEPT_EULA=YES``。构建物理仿真 Docker 镜像时选择 ``BUILD_TARGET=embodied-robodojo``。当仓库变量 ``ROBODOJO_ASSETS_ROOT`` 指向 embodied runner 的资产目录时，CI 才会运行仿真 smoke job。

``env/robodojo_put_bottles_into_dustbin.yaml`` 选择物理接口，默认 native 顺序为 [左臂，右臂，左夹爪，右夹爪]；ALOHA policy 应设置 ``state_action_order: aloha``。保持 ``enable_offload: False``，关闭仿真器会永久释放资源。reset 重试返回尝试次数与实际布局 ID。历史 JSON 字段 ``success_seeds`` 存放布局 ID，并非测得的 policy 成功样本。CPU 测试使用外部 runtime 替身验证适配接口；真实仿真和完整 OpenSora CUDA 运行需要准备资产与 checkpoint 后单独验证。
