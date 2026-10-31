\# RoboTwin 2.0 仿真接入实验记录



\## 研究目标

探索如何将 RoboTwin 2.0 接入 RLinf 强化学习框架。



\## 当前进度

\- 已成功克隆 RLinf 仓库

\- 已创建个人开发分支 feat/robotwin-baixidyt

\- 已了解 Git 基本操作



\## 遇到的问题

暂无



\## 下一步计划

1\. 阅读 RLinf 项目结构

2\. 了解 RoboTwin 2.0 仿真环境

3\. 尝试运行官方评测示例

## RoboTwin 接入现状



目前已经确认 RLinf 仓库中存在 RoboTwin 仿真环境适配代码，路径为：



`rlinf/envs/sim/robotwin/`



其中核心文件为：



`robotwin\_env.py`



核心环境类为：



`RoboTwinEnv`



该类继承自 Gymnasium 的 `gym.Env`，负责连接 RLinf 与底层 RoboTwin 仿真环境。



目前已确认的关键接口包括：



\* `reset()`：重置 RoboTwin 仿真环境，并获取初始 observation。

\* `step(actions)`：向 RoboTwin 发送动作，并接收新的 observation、reward、termination、truncation 和额外信息。

\* `\_extract\_obs\_image()`：将 RoboTwin 返回的原始 observation 整理为 RLinf 使用的数据格式。

\* `\_record\_metrics()`：记录任务成功率、累计 reward 和 episode 长度等指标。

\* `chunk\_step()`：支持一次向环境发送一段连续动作。



在 `\_init\_env()` 中，RLinf 会导入：



`robotwin.envs.vector\_env.VectorEnv`



这说明 RLinf 中的 `RoboTwinEnv` 主要起适配层作用，底层仿真执行仍由 RoboTwin 自身的 `VectorEnv` 完成。



当前结论：



RLinf 已经具备 RoboTwin 的基础环境适配，因此现阶段的主要任务不是从零编写 RoboTwin 环境接口，而是复现现有评测流程，确认环境依赖、模型、资产和配置是否能够正常运行。



下一步计划：



1\. 在实验室 Linux + NVIDIA GPU 服务器上检查运行环境。

2\. 安装 RoboTwin 相关依赖。

3\. 准备 RoboTwin 仿真资产和预训练模型。

4\. 尝试运行一个最小 RoboTwin evaluation。

5\. 记录成功率或具体报错信息。



## RoboDojo 接入初步调研

RoboDojo 当前官方发布以 evaluation 为主。RoboDojo 本身负责仿真客户端、任务、资产、配置以及评测结果生成；policy 的集成与 policy server 由 XPolicyLab 负责。

官方仓库中的关键结构包括：

* `env/`：仿真环境底层与管理模块
* `env_cfg/`：仿真、场景、机器人和相机配置
* `task/RoboDojo/`：RoboDojo 任务逻辑与任务配置
* `scripts/robodojo.sh`：RoboDojo 官方评测入口
* `scripts/eval_policy.sh`：由 XPolicyLab 调用的仿真客户端入口
* `XPolicyLab/`：policy server 与不同模型的集成

RoboDojo 的典型评测流程可以理解为：

模型 / Policy
→ XPolicyLab policy server
→ RoboDojo simulator client
→ RoboDojo 仿真任务
→ 返回 observation、动作执行结果和评测指标

因此，RLinf 接入 RoboDojo 的潜在重点不是首先重写一套仿真环境，而是研究 RLinf 当前模型推理接口如何适配 XPolicyLab 的 policy server 接口。

下一步需要继续确认：

1. XPolicyLab 对 policy server 的接口要求
2. observation 的数据格式
3. action 的数据格式
4. RLinf 当前模型输出是否可以直接转换为该 action 格式



## RoboDojo / XPolicyLab 接口要求

根据 RoboDojo 与 XPolicyLab 官方仓库，RoboDojo 当前为 eval-only，policy 的集成和 policy server 由 XPolicyLab 负责。

一个 XPolicyLab policy 至少需要提供：

* `eval.sh`
* `deploy.yml`

其中：

* `eval.sh` 用于启动 policy 的评测流程
* `deploy.yml` 用于声明 policy server 的 host、port、action mode 和 policy 运行配置

RoboDojo 的典型单机评测流程为：

RoboDojo CLI
→ 启动 XPolicyLab policy server
→ 启动 RoboDojo simulator client
→ client 将 observation 发送给 policy server
→ policy server 返回 action
→ RoboDojo 执行动作并继续评测

默认通信方式为 WebSocket。

因此，RLinf 接入 RoboDojo 的一个可行方向是：

为 RLinf 模型实现一个 XPolicyLab policy adapter，使 RLinf 的模型能够通过 XPolicyLab policy server 接收 observation，并返回 RoboDojo 所需的 action。

当前还需要继续确认：

1. observation 的具体字段
2. action 的具体维度与格式
3. RLinf 当前模型输入输出如何转换
4. 是否已有相近的 XPolicyLab policy adapter 可以参考



