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



