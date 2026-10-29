# 本地适配改动盘点

本文记录 RLinf 工作区对 RoboTwin 2.0、RoboDojo 和 OpenSora 世界模型接口的改动，以及截至 2026-10-29 的验证边界，供审查和服务器部署参考。它是当前未提交工作区的快照，不替代面向用户发布的中英文 Sphinx 文档。

盘点基线：分支 `feature/robodojo-adaptation`，HEAD `e1a82ebf0c1a`。编写本文前有 12 个已跟踪文件修改、13 个未跟踪文件；本文本身也是新增文件。

## 环境与动作接口

RLinf 对接物理环境时，需要将 policy 输出的动作转换为环境所需顺序，接收观测和结束信号，并正确管理环境生命周期。当前工作区新增 RoboDojo 适配，同时扩展已有世界模型路径。

| 文件 | 修改内容与作用 |
| --- | --- |
| [rlinf/envs/sim/robodojo/robodojo_env.py](rlinf/envs/sim/robodojo/robodojo_env.py)、[rlinf/envs/sim/robodojo/__init__.py](rlinf/envs/sim/robodojo/__init__.py) | 增加 RoboDojo 环境包装，处理外部运行时、复位、步进、观测、成功/截断和关闭等接口；运行时桥接依赖需另行准备。 |
| [rlinf/envs/__init__.py](rlinf/envs/__init__.py) | 注册 RoboDojo 环境类型。 |
| [rlinf/envs/action_utils.py](rlinf/envs/action_utils.py) | 增加 RoboTwin、RoboDojo 相关动作处理，并保留已有环境的处理路径。RoboDojo 双臂 14 维动作需区分 native 和 ALOHA 顺序。 |
| [rlinf/envs/sim/world_model/backend/opensora.py](rlinf/envs/sim/world_model/backend/opensora.py) | 扩展 OpenSora 世界模型后端，检查目标域动作维度和统计信息，支持按目标数据统计量归一化。 |
| [rlinf/envs/sim/world_model/env.py](rlinf/envs/sim/world_model/env.py) | 扩展世界模型环境的配置与接口处理；当前目标配置以图像预测为主，不提供已验证的物理成功判断或奖励模型。 |

RoboDojo 的 native 顺序为左右臂后接左右夹爪，ALOHA 顺序为左臂及左夹爪后接右臂及右夹爪。样本里的 `state_target` 尚不能直接视为已验证的执行命令；真实运行前必须核对语义和排列。

## 首任务配置与运行环境

| 文件 | 修改内容与作用 |
| --- | --- |
| [examples/embodiment/config/env/opensora_robotwin_place_empty_cup.yaml](examples/embodiment/config/env/opensora_robotwin_place_empty_cup.yaml)、[examples/embodiment/config/env/opensora_robodojo_put_bottles_into_dustbin.yaml](examples/embodiment/config/env/opensora_robodojo_put_bottles_into_dustbin.yaml) | 为两个首任务增加 OpenSora 世界模型环境配置。 |
| [examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml](examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml) | RoboDojo 物理环境的首任务配置。 |
| [evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml](evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml)、[evaluations/robodojo/put_bottles_into_dustbin_seeds.json](evaluations/robodojo/put_bottles_into_dustbin_seeds.json)、[evaluations/run_eval.sh](evaluations/run_eval.sh) | 增加 RoboDojo 评估配置、种子和运行入口；配置存在不代表评估已成功运行。模型、统计量、资产及运行时路径需在目标机器准备。 |
| [requirements/install.sh](requirements/install.sh)、[requirements/embodied/envs/robodojo-override.txt](requirements/embodied/envs/robodojo-override.txt) | 增加 RoboDojo 依赖安装和版本约束；目标物理运行环境以 Linux、NVIDIA GPU 和 Isaac Sim 相关依赖为前提，需按安装脚本配置。 |
| [docker/Dockerfile](docker/Dockerfile) | 增加 RoboDojo 相关镜像构建目标；该文件保留为项目构建选项，本机真实仿真验证不因此要求使用 Docker。 |

## 测试、持续集成和文档

| 文件 | 修改内容与作用 |
| --- | --- |
| [tests/unit_tests/test_robodojo.py](tests/unit_tests/test_robodojo.py)、[tests/unit_tests/test_world_model.py](tests/unit_tests/test_world_model.py) | 覆盖 RoboDojo 包装、动作处理与世界模型接口等行为。 |
| [tests/robot_mocks/robodojo_runtime/__init__.py](tests/robot_mocks/robodojo_runtime/__init__.py)、[tests/robot_mocks/robodojo_runtime/bridge.py](tests/robot_mocks/robodojo_runtime/bridge.py) | 为单元测试提供 RoboDojo 运行时替身；它不是 Isaac Sim 物理环境。 |
| [tests/e2e_tests/embodied/robodojo_env_smoke.py](tests/e2e_tests/embodied/robodojo_env_smoke.py) | 提供真实运行时的 smoke 测试入口；需满足资产、运行时及 GPU 环境条件。 |
| [.github/workflows/ci-tests.yml](.github/workflows/ci-tests.yml)、[.github/workflows/docker-build.yml](.github/workflows/docker-build.yml)、[.github/workflows/embodied-e2e-tests.yml](.github/workflows/embodied-e2e-tests.yml) | 调整 CI 路径、镜像构建及 RoboDojo 端到端测试配置；这些变更不表示远端 CI 已执行通过。 |
| [docs/source-en/rst_source/examples/embodied/opensora.rst](docs/source-en/rst_source/examples/embodied/opensora.rst)、[docs/source-zh/rst_source/examples/embodied/opensora.rst](docs/source-zh/rst_source/examples/embodied/opensora.rst) | 更新英文和中文 OpenSora 示例文档。 |

## 已有验证与未完成事项

- Windows 原生 CPU 接口回归记录为 **64 passed、2 deselected**；未选项依赖 Linux Bash 路径，部分测试使用运行时替身和小模型。此前 Docker CPU 回归记录为 66 passed。本文编写时未重新运行这些测试。
- RoboTwin 的 SAPIEN 真实任务曾完成短轨迹 smoke，但使用软件渲染及本地降噪兼容设置；这不等于 OpenSora 与真实物理环境的模型闭环验收。Windows 原生 SAPIEN 基础渲染和物理能力曾单独验证，完整任务尚未在原生环境验收。
- RoboDojo 的 Windows 原生 Isaac Sim 在 RTX 场景插件发生 `0xC0000005`，未完成任务步进和相机验证；因此尚不能宣称 RoboDojo 真实物理接口已验收。
- 尚需在服务器验证两套真实环境与 RLinf/OpenSora 的端到端接口、完整目标域权重及 CUDA 训练、奖励模型、留出集表现和物理任务成功率。配置文件或单元测试通过均不能替代上述验收。

详细日志和实验记录位于同级工作区的 `experiments/本机数据与接口复核报告.md`、`experiments/Windows原生验证报告.md` 和 `experiments/真实仿真接口测试报告.md`；这些文件不属于本仓库。
