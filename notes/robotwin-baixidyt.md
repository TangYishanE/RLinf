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


## RoboDojo observation 调研

RoboDojo 的 observation 由 `ObsManager` 负责组织。

关键文件：

`env/observation_manager/obs_manager.py`

在评测环境 `src/eval_client/eval_env.py` 中，会初始化 `ObsManager`，并在仿真过程中调用其 `get_obs()` 获取 observation。

因此，下一步需要重点阅读 `ObsManager.get_obs()`，确认最终返回给 policy server 的 observation 字段和格式。



## RoboDojo observation 结构

根据 `env/observation_manager/obs_manager.py` 中的 `ObsManager.get_obs()`，RoboDojo 会为每个环境构造 observation 字典，主要字段包括：

- `vision`：相机数据，例如 RGB、深度、图像尺寸以及可选的相机内外参
- `state`：机器人状态，例如关节状态和末端执行器位姿
- `action`：当前或上一控制相关的数据
- `instruction`：任务自然语言描述
- `data_format_version`
- `additional_info`

这说明 RoboDojo 的 observation 与 RLinf RoboTwin 的输入结构存在较强对应关系：

- RoboDojo `vision` ↔ RLinf `main_images / wrist_images`
- RoboDojo `state` ↔ RLinf `states`
- RoboDojo `instruction` ↔ RLinf `task_descriptions`

因此，一个潜在的 RLinf → RoboDojo adapter 主要需要解决 observation 字段映射、图像维度转换以及 action 格式转换。

## RoboDojo action 格式初步结论

RoboDojo / XPolicyLab 当前支持的 action type 至少包括：

- `joint`
- `ee`

其中：

- `joint` 表示基于机器人关节状态的动作
- `ee` 表示基于末端执行器（end-effector）位姿的动作

RoboDojo 会根据 action 字典中的 key 判断动作类型。例如，如果 action 中存在机器人对应的关节 key，则识别为 `joint`；如果存在类似 `left_ee_pose` / `right_ee_pose` 的 key，则识别为 `ee`。

XPolicyLab 的评测参数中也明确包含 `action_type`，通常为 `joint` 或 `ee`。

因此，RLinf 接入 RoboDojo 时，需要重点确认：

1. RLinf 当前模型输出的是 joint action 还是 ee action
2. action 的具体维度、顺序和单位
3. 是否需要把 RLinf 的 action 转换成 RoboDojo / XPolicyLab 所要求的 action 字典

## RLinf RoboTwin action 结构

根据 `rlinf/envs/sim/robotwin/robotwin_env.py`，RoboTwin 环境中的 action 最后一维为 14：

`[num_envs, horizon, 14]`

RLinf 官方文档也说明，RoboTwin 使用 ALOHA 风格的双臂连续 action chunk；在 OpenPI + RoboTwin 配置中，action_dim 为 14。

因此目前可以确认：

* RoboTwin action 为 14 维
* action 用于双臂连续控制
* 模型通常一次可以输出一个 action chunk，而不是只输出单步动作

当前还不能仅凭 `robotwin_env.py` 完全确认 14 个维度在所有模型配置中的具体物理含义，需要继续查看具体 policy / model 的 action adapter 与配置。

与 RoboDojo 对接时，下一步需要判断：

* 这 14 维是否可以直接映射到 RoboDojo 的 `joint` action
* 或者是否需要转换成 RoboDojo 的 `ee` action




## RoboTwin 14 维 action 的具体含义

对于 RLinf 的 OpenPI + RoboTwin ALOHA 配置，14 维 action 可以进一步确定为双臂 joint + gripper 控制。

具体结构为：

- 第 0~5 维：左机械臂 6 个关节
- 第 6 维：左夹爪
- 第 7~12 维：右机械臂 6 个关节
- 第 13 维：右夹爪

即：

左臂 6 joint + 1 gripper
+
右臂 6 joint + 1 gripper
=
14 维

OpenPI 的 ALOHA policy 同样使用 14 维 state 和 14 维 action。

因此，RLinf 的 OpenPI RoboTwin action 在控制类型上与 RoboDojo 的 `joint` action 有较强对应关系。

下一步需要继续确认：

1. RoboDojo joint action 的具体 key
2. RoboDojo 左右臂关节顺序
3. action 单位是否一致
4. gripper 数值范围是否一致





## RoboDojo joint action 结构

RoboDojo 的 policy action 不是简单的裸向量，而是字典结构。

对于双臂机器人，action 会分别包含左右机械臂和夹爪相关字段，并使用 `left_` / `right_` 前缀区分两侧。

RoboDojo 在执行 joint action 时，会：

1. 读取每个 target robot 的 arm action
2. 读取对应的 gripper action
3. 将 gripper 输入裁剪到合法范围
4. 根据机器人配置中的 gripper_scale 等参数转换为真实控制值
5. 最终生成底层机器人控制命令

因此，RLinf 的 RoboTwin 14 维 action 不能直接原样传入 RoboDojo。

一个潜在 adapter 可以执行类似转换：

RoboTwin 14-D vector
→ 左臂 6 joint
→ 左 gripper
→ 右臂 6 joint
→ 右 gripper
→ RoboDojo joint action dict

需要继续确认：

- RoboDojo 当前机器人配置生成的具体 key 名
- 左右臂关节顺序是否与 ALOHA 完全一致
- joint 数值单位
- gripper 输入范围



## RoboDojo joint / gripper 数值范围

根据 RoboDojo 的 `eval_env.py`：

### Gripper

RoboDojo 会先将 policy 输出的 gripper 数值裁剪到：

`[0, 1]`

随后再根据具体机器人的：

- `gripper_scale`
- `gripper_move`
- mimic joint 配置

把归一化的 gripper 输入转换为底层真实夹爪位置。

因此，RLinf → RoboDojo adapter 需要确保夹爪输出能够正确映射到 `[0, 1]`。

### Arm joint

RoboDojo 的 joint action 会作为目标关节 `position` 传给底层控制逻辑，并在当前关节状态与目标状态之间进行插值。

目前可以确认：

- action 表示目标 joint position
- RoboDojo 会在 joint space 中做插值

目前尚未完全确认：

- arm joint position 的单位是否统一为 radians
- 是否存在 policy 侧额外 normalization
- RLinf RoboTwin 的 joint 数值范围是否与 RoboDojo 完全一致




## RoboDojo ARX X5 双臂 joint 结构

RoboDojo 的 ARX X5 双臂机器人每侧包含 6 个 arm joints。

在相关实现中，arm joint 会按：

- joint1
- joint2
- joint3
- joint4
- joint5
- joint6

的顺序排列。

双臂整体结构为：

左臂 6 joint + 左 gripper
+
右臂 6 joint + 右 gripper
=
14 维

因此，从 action 维数结构来看：

RLinf RoboTwin ALOHA 14-D action
与
RoboDojo ARX X5 14-D joint + gripper action

具有较强对应关系。

但是，目前仍不能直接认为数值可以一一复制，因为还需要继续确认：

1. 各 joint 的物理零位定义
2. joint limit
3. 单位
4. gripper 开合方向
5. normalization 方式



## RoboDojo X5 joint limit 与单位

目前可以确认：

- ARX X5 的 arm joint 为 revolute joint
- RoboDojo 会从自身 Assets 中的 X5 URDF 读取每个 joint 的 lower / upper limit
- RoboDojo 的 action spec 会直接使用这些 URDF limit 作为关节动作上下限
- 对于 revolute joint，URDF 中的关节角按照 ROS/URDF 约定使用 radians

ARX 官方文档另外给出了便于阅读的角度范围，例如：
- J1：约 -150° ~ 180°
- J2：约 0° ~ 210°
- J3：约 0° ~ 180°
- J4：约 -90° ~ 90°
- J5：约 -90° ~ 90°
- J6：约 -120° ~ 120°

需要注意：

外部旧版 X5 URDF 中可能存在较宽松或占位性质的 joint limit，因此真正实现 RoboDojo adapter 时，应以 RoboDojo 仓库 `Assets/Robots/x5/X5A.urdf` 中的限制为准。

当前阶段结论：

RLinf RoboTwin 的 14-D ALOHA joint+gripper action
与
RoboDojo ARX X5 的 14-D joint+gripper action

在维度结构上可以对应，但数值映射时必须根据 RoboDojo 自身 URDF 的 joint limit 和 gripper [0,1] 范围进行检查。




## RoboTwin ↔ RoboDojo 接入对照

### 1. 环境接入方式

**RLinf + RoboTwin**

* RLinf 内已经存在 `RoboTwinEnv`
* 由 RLinf 直接调用 RoboTwin 环境

**RoboDojo**

* 采用 RoboDojo simulator client
* policy 侧通过 XPolicyLab policy server 提供

**当前判断**

* RoboDojo 更可能需要额外 adapter，而不是重写整个仿真环境

### 2. Observation

**RLinf + RoboTwin**

* 图像：`main_images`、`wrist_images`
* 状态：`states`
* 任务文本：`task_descriptions`

**RoboDojo**

* 图像：`vision`
* 状态：`state`
* 任务文本：`instruction`

**当前判断**

* 三类核心输入都可以做字段映射

### 3. Action

**RLinf + RoboTwin**

* 14-D 双臂连续 action
* 左 6 joint + gripper
* 右 6 joint + gripper

**RoboDojo**

* 支持 `joint` / `ee` action
* ARX X5 双臂同样是左右各 6 joint + gripper
* joint action 以字典形式提供

**当前判断**

* 两边维度结构高度一致
* 但需要把 RLinf 的 14-D vector 转换成 RoboDojo action dict

### 4. Joint 与 Gripper

**RLinf + RoboTwin**

* ALOHA 风格 joint + gripper
* joint limit 由 RoboTwin / ALOHA 配置决定

**RoboDojo**

* joint action 表示目标 joint position
* joint limit 从 X5 URDF 读取
* gripper policy 输入会裁剪到 `[0, 1]`

**当前判断**

* 控制语义接近
* 仍需检查 joint 顺序、单位、零位、limit 和 gripper 开合方向

### 5. Policy 通信与评测

**RLinf + RoboTwin**

* RLinf 内部直接调用模型和环境
* 使用 RLinf evaluation pipeline

**RoboDojo**

* 通过 WebSocket policy server 通信
* 使用 RoboDojo CLI / XPolicyLab 评测流程

**当前判断**

* 需要把 RLinf policy 封装成 XPolicyLab 可调用的 policy server

## 当前阶段结论

目前最可能的 RoboDojo 接入方式不是重写 RoboDojo 仿真器，而是在 RLinf 与 XPolicyLab 之间增加一层 adapter。

该 adapter 至少需要负责：

1. 将 RoboDojo observation 转换为 RLinf 模型所需输入格式
2. 调用 RLinf policy 推理
3. 将 RLinf 输出的 14-D action 拆分并转换为 RoboDojo joint action dict
4. 处理 gripper `[0, 1]` 映射
5. 检查 joint 顺序、单位、joint limit 与零位定义
6. 通过 XPolicyLab policy server 返回 RoboDojo 可执行动作





