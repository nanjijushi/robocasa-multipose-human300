# 完整移动方案对照：纯 LoRA 动作头与独立外挂

这里比较两套方案“移动调整位置，然后让原 Pi0.5 操作抽屉”的效果。外挂输出方向及停止判断；LoRA 直接生成底盘动作。两边都提供原始训练器、推理服务、仿真客户端和权重，供服务器复现实验。它们不是当前训练的好/坏视角 Adapter。

## 选用版本

| 对照方案 | 权重 | 选择依据 |
|---|---|---|
| 纯 LoRA 动作头 | `pure_mobility_lora_v2.npz` | 历史 `run_seed25_pure_mobility_lora_v2_server.sh` 明确使用的纯动作头基线；22,118,400 个 LoRA 参数，仅监督底盘 7/8/9 维 |
| 后期独立外挂 | `external_joint_reposition_v1.pt` | `joint_offline_value_reposition_v1`，2026-10-08 训练，最佳 step 700；839,758 参数，价值、六方向动作和停止三分支 |

服务文件时间线显示，之后的 v12–v20 转向 VLA 融合/内部阶段判断，因此这里选的是后期**独立外挂**。没有足够跨版本闭环统计证明它是所有版本的最佳；不能用离线 loss 宣称它更强。

LoRA v5 是接收评分反馈的版本，不属于这里的纯 LoRA 基线。评分器 v6 也不是完整外挂移动模块，均未混入这两组。

`weights_manifest.json` 固定来源及 SHA256；`*_metrics.json` 是历史训练记录；`source_provenance.json` 记录原文件哈希和打包改动。训练日志里出现的旧服务器路径仅作追溯。LoRA 历史训练报告提到评分停止，但本次使用后来明确的 **pure runtime** 启动配置：无评分器、无评分反馈。

## 在另一台服务器取得代码与权重

如果已有本仓库，进入它执行 `git pull --ff-only`；有本地修改时请另建下面的对照目录，不覆盖正在用的训练代码。

```bash
cd /wangxinhao/tmp
git clone --recurse-submodules https://github.com/nanjijushi/robocasa-multipose-human300.git robocasa-mobility-comparison
cd /wangxinhao/tmp/robocasa-mobility-comparison
source /root/miniforge/etc/profile.d/conda.sh
conda activate /wangxinhao/envs/robocasa
export PYTHON_BIN=/wangxinhao/envs/robocasa/bin/python
python comparison_models/download_weights.py
```

依赖沿用仓库根 README 的环境安装、RoboCasa 资源和 OpenPI 兼容补丁。`tested_environment.json` 记录源服务器版本，不建议在已有可用环境里盲目重装全部依赖。基础 Pi0.5 权重仍使用原始 Human300 multitask/75000，不随 Release 重复发布：

```bash
export PI05_CHECKPOINT=/wangxinhao/tmp/robocasa-multipose-human300/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

若复用已安装、已打补丁的 OpenPI，可显式 `export OPENPI_ROOT=/你的/openpi目录`。目录包含 `src` 和 `packages/openpi-client/src`。

## 两套完整运行命令

同一张 GPU 上顺序运行两套方案。先开服务终端：

```bash
# A：纯 LoRA 动作头
bash comparison_models/run_server.sh lora
# 完成 A 并退出服务后，运行 B：后期独立外挂
bash comparison_models/run_server.sh external
```

另一个终端激活同一环境、进入同一仓库后，启动对应客户端：

```bash
bash comparison_models/run_client.sh lora --seed 25 --placement-seed 2551
# B 必须使用相同的场景 seed 与 placement-seed
bash comparison_models/run_client.sh external --seed 25 --placement-seed 2551
```

客户端沿用原交互式 MuJoCo viewer，需要服务器桌面/可用显示环境。按 `v` 启动移动及后续操作，`x` 回退，`q` 退出。它不是已经完成的无界面批量性能评测器。`--attempt-max-steps 600` 在历史客户端中是提示上限，**不是强制截断预算**，不要据此宣称双方已按严格 600 步完成公平评测。

`run_server.sh ... --preflight` 只检查路径/权重结构，不能替代真实 GPU 推理。服务默认均为 5 次 flow 去噪步；旧纯 LoRA 启动器用 3 次，如复现旧设置可加 `--num-steps 3`，公平比较时两组保持相同。

可用下面的命令检查真实权重的 GPU 推理接口。它使用合成观测，不启动仿真、不会移动机器人，也不衡量性能：

```bash
source comparison_models/env.sh
"$PYTHON_BIN" comparison_models/smoke_inference.py --mode lora --base-checkpoint "$PI05_CHECKPOINT"
"$PYTHON_BIN" comparison_models/smoke_inference.py --mode external --base-checkpoint "$PI05_CHECKPOINT"
```

## 动作与停止究竟由谁控制

- **LoRA**：当前主相机、腕部图像和状态进入 action expert，输出底盘命令；运行时锁机械臂、裁剪/死区处理底盘动作，按动作强度和连续停止规则交接原 Pi0.5。未加载外挂评分器。
- **外挂**：冻结主相机空间特征、任务嵌入、底盘 x/y/yaw 进入独立网络；客户端取六方向最大概率，执行 4 个控制步后再观测。停止概率 ≥0.5 时暂停，连续 3 次确认后交接原 Pi0.5。
- 外挂历史兼容服务仍初始化一个 LoRA 模型以复用图像编码及操作交接框架；本包让它读取同一份 v2 权重。**外挂移动分支只使用独立网络方向/停止输出，不调用 LoRA 生成的动作。** LoRA 仅修改 action expert，不修改外挂使用的冻结图像编码器。
- 两套操作阶段都切回原基础模型并锁底盘。外挂客户端的 `score_only=True` 是历史接口名，实际返回价值、动作方向、停止三种输出，不能把它误解为只打分。

## 如何比较性能

在固定、独立留出的场景与起始状态上比较：最终目标抽屉成功率、移动结束后操作成功率、未能停止比例、移动控制步数/位移及总耗时。确认目标抽屉和真实 rollout 提示词一致；按人工核实的目标成功判定，不把抓错抽屉算成功。

交互演示只是原运行方式。正式批量比较需在同一评测器中恢复**同一 XML、完整仿真初始状态、提示词和随机种子**，给两组相同的移动与操作预算，并记录预算耗尽为失败/超时。仅场景 seed 一样不足以保证相同起点；即使用了相同 placement-seed，也要核对起始状态/图像。训练涉及 seed17/19/31，应单独报告已见和未见场景。

两套训练集、监督目标和观测输入并不完全相同，因此这首先是**历史完整方案对照**，不能把所有差距归因于“外挂还是 LoRA”这一单一因素。没有用本次上传跑出新的性能结论。

## 配套训练代码

- `src/train_seed17_pi05_implicit_mobility_v1.py`：纯 action-expert LoRA 训练器，v2 使用 `--balanced-v2`（包括 yaw 与停止帧采样）；详情见 `--help` 和历史 metrics。
- `src/train_joint_offline_value_reposition_v1.py`：独立价值/方向/停止联合训练器。
- `src/extract_three_scene_spatial_4x4_embeddings_v1.py`：冻结视觉特征提取。

训练需要原 A→B 轨迹、索引与任务嵌入，本次没有发布私人数据。请显式传入数据路径和新的输出目录；上传代码和最终权重不等于已具备从零逐位复现全部训练的数据。本次也不修改当前 Adapter 训练流程。
