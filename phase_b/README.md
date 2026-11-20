# LoMAP 实验 B · Maze2D 本机复现

阶段 1 验证 `uv`、PyTorch CUDA、D4RL 数据、无界面 MuJoCo 和 CPU Faiss。阶段 2 在 U-Maze 短训，重载一个 checkpoint 做 LoMAP 开关对照，再用 MuJoCo 执行计划。阶段 3–4 在 Maze2D-Medium 训练并测量生成计划穿墙，使用论文 Figure 2 和 §4.1 的实验问题。 [LoMAP 论文](https://proceedings.mlr.press/v267/lee25f.html)

## 环境与来源

- `uv.lock` 固定 Python 3.13 的 Linux x86-64 环境。首个成功试跑使用 PyTorch 2.15.0+cu132、MuJoCo 3.3.7 和 Faiss CPU 1.12.0。CUDA 用户态库由 Python 环境提供，本机 NVIDIA 驱动负责运行。依赖走清华 PyPI 镜像，试跑时复用了本机已经缓存并与锁文件哈希匹配的 CUDA wheels。
- 模型、DDPM 采样器、LoMAP 的邻居重加噪和 PCA 使用作者发布代码，固定到 `8ccb052481484352affaada26a0895102aef3e52`。本地运行入口替换 GPU Faiss 为 CPU IVF-Flat。Maze2D 配置的 value guidance 权重为 0，本次按这个设置跳过未参与采样的 classifier。U-Maze 的 horizon 和 diffusion steps 都取 128，扩散步数按论文附录 E.2。 [官方源代码](https://github.com/leekwoon/lomap/tree/8ccb052481484352affaada26a0895102aef3e52) · [PyTorch 版本依据](https://pytorch.org/blog/pytorch-2-14-release-blog/)
- [`prepare_assets.py`](prepare_assets.py) 从固定版本的 D4RL PointMaze 代码生成 U-Maze 和 Medium 的 MuJoCo XML。无界面适配器沿用原模型、控制与速度裁剪及稀疏奖励，调用官方 `mujoco` Python API。当前 smoke rollout 以数据窗口的真实起点重置环境，执行 127 步 waypoint 控制。 [PointMaze 源码](https://github.com/leekwoon/lomap/blob/8ccb052481484352affaada26a0895102aef3e52/d4rl/pointmaze/maze_model.py)

## 数据处理与配对比较

1. 从 Berkeley 下载原始 `maze2d-{umaze,medium}-sparse-v1.hdf5` 后记录 SHA-256。读取 `observations`、`actions` 和 `terminals`，检查观测和动作维度为 4 和 2，数值均有限。原始文件留在 `data/`。 [D4RL 数据入口](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py)
2. 沿用作者的切分规则。到达固定 goal 的 0.5 半径内后，在下一步离开半径的地方结束片段。原始 terminal 也保留为边界。每条训练窗口完全落在一个片段内。四个 observation 通道用全数据的 min/max 映到 `[-1, 1]`，动作沿用原值。代码只存连续数组和片段索引，通过索引惰性取窗口。 [发布版 preprocessing](https://github.com/leekwoon/lomap/blob/8ccb052481484352affaada26a0895102aef3e52/cleandiffuser_ex/dataset/d4rl_maze2d_dataset.py#L42-L113)
3. 试跑按固定种子抽 8,192 条 U-Maze 窗口，建立 64 个聚类的 CPU IVF-Flat 索引。检索沿用官方 cosine 权重，端点位置权重 100、动作权重 0。LoMAP 取 10 个邻居，按 0.99 累计解释方差投影。保留了 8 组起点、共同目标、两组采样及 MuJoCo 执行。两组使用同一 EMA checkpoint、同一个 reverse DDPM 随机流，LoMAP 的邻域噪声单独抽样。

## 独立运行

在本机终端执行以下命令即可启动。源码和数据已就位时，从 `run_local.sh` 开始即可。

```bash
cd phase_b
mkdir -p .upstream data
git clone https://github.com/leekwoon/lomap.git .upstream/lomap
git -C .upstream/lomap checkout --detach 8ccb052481484352affaada26a0895102aef3e52
aria2c --dir="$PWD/data" https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5
aria2c --dir="$PWD/data" https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5
uv sync --locked
uv run --locked python prepare_assets.py
```

```bash
# 可用已缓存的 wheels；虚拟环境仍固定在 phase_b/.venv。
run_dir=$(bash run_local.sh umaze)
cat "$run_dir/unit.txt"
tail -f "$run_dir/run.log"
# 任务结束后读退出状态，0 表示成功。
cat "$run_dir/exit_code"
```

`run_local.sh` 使用 Linux systemd 用户服务在后台运行任务。每次运行写入专用目录，可在另一个终端看 `systemctl --user status "$(cat "$run_dir/unit.txt")"`。机器休眠会暂停运行，重启会终止进程。`bash run_local.sh env` 可单独安装依赖。

## 阶段 1–2 实测

| 检查 | 实测结果 |
|---|---|
| U-Maze 数据 | 100 万条 transitions，801,704 个合法的 128 步窗口 |
| Medium 数据 | 200 万条 transitions，1,743,301 个合法的 256 步窗口 |
| U-Maze 训练 | 600 步，batch 64，耗时 8.53 秒。去掉前 20 步后，平均 13.78 ms/步，P90 为 14.13 ms/步 |
| 显存 | 训练时 PyTorch 峰值分配约 184 MiB、峰值保留 224 MiB。`nvidia-smi` 每 200 ms 观测到整卡从 26 MiB 空闲基线最高升到 388 MiB |
| CPU Faiss | 8,192 条窗口建索引约 0.26 秒。索引通过已知向量自检 |
| U-Maze 配对采样 | 每组 8 条 plan，128 个采样步。关 LoMAP 约 0.24 秒，开 LoMAP 约 0.53 秒。Faiss 调用分别为 0 和 64 次 |

两份 HDF5 的切分计数与独立计算一致。各检查了首、中、尾窗口，observation 与作者 `MinMaxNormalizer` 相同，action 与原始切片相同。U-Maze 中向墙连续施加控制后观察到 MuJoCo 接触，球停在墙前。配对运行两组端点完全一致，reverse DDPM 随机流状态一致，且 LoMAP 改变了生成轨迹。GPU 同 seed 复跑通过同样的流程检查，模型权重和计划有小幅数值差异。

运行产物写入 `runs/`，其中可能包含机器环境信息、日志和模型权重；此目录已加入忽略规则，不应提交。

## 阶段 3–4 · Medium

**训练。** 从头训练作者发布的 Janner U-Net，使用前面说明的固定 D4RL Medium 数据、256 步 horizon、256 diffusion steps、batch 64、seed 20261120。遍历每轮时对合法窗口重新洗牌，最后不足一个 batch 的窗口留待后续轮次抽取。用作者配置的 AdamW、EMA 0.9999 和 cosine 学习率调度，短训预算为 50,000 个 optimizer steps。保存普通模型和 EMA 模型，并另存可续训的模型、optimizer、scheduler、CUDA 随机流和已完成步数，每 1,000 步原子替换一次。数据顺序由 seed 和 epoch 编号重建。 [训练入口](medium_train.py) · [作者训练入口](https://github.com/leekwoon/lomap/blob/8ccb052481484352affaada26a0895102aef3e52/pipelines/lomapdiffuser_d4rl_maze2d.py#L37-L115)

**评估。** 从 Medium 地图的可通行格子中按固定种子无放回抽取 20 对不同的有序格子。在各格子远离边界的区域内抽取真实离线状态作为起、终点，端点包含位置和速度。对每对起终点各生成 10 条轨迹，两组共用训练完成的 EMA checkpoint、conditioning 和 DDPM 随机流。LoMAP 邻域重加噪使用独立子流。用同一份原始数据无放回抽取 65,536 个合法窗口建 CPU cosine IVF-Flat 索引，`nlist=256`、`nprobe=5`。LoMAP 按 Medium task 配置仅在 `proj_range=[0.3, 0.4]` 投影，10 个近邻、`tau=0.99`，value guidance 权重为 0。 [评估入口](medium_evaluate.py) · [作者 Medium 配置](https://github.com/leekwoon/lomap/blob/8ccb052481484352affaada26a0895102aef3e52/configs/lomapdiffuser/maze2d/task/maze2d-medium-v1.yaml)

**穿墙判定。** 从原版 MuJoCo XML 读取 38 个矩形墙体，把 world 平移到 D4RL observation 的 qpos 坐标系。在反归一化后，逐段检查相邻位置的直线插值是否进入墙体内部，使用 `1e-7` 边界容差。切线和仅碰到边界的点排除在计数外。按论文 §4.1，对每对起终点，只要前 `n` 条轨迹中至少有一条触发穿墙，该对就计入图中的 artifact ratio。另报全部 200 条计划中的逐轨迹比例。本口径用粒子中心几何判定，未计粒子半径、动力学可达性或实际控制成功率。已用穿越、切线、端点和反向线段验证判定，并检查固定抽取的 512 个离线窗口未触发穿墙。

**范围。** 本机跑的是 Figure 2 左图的小规模同逻辑对照，采用 50,000 步短训、缩小的 CPU 检索库、自行规定的端点与几何 oracle、最多 10 次采样，并比较 Diffuser 和 Diffuser + LoMAP 两组。原图包含 RGG 且最多 30 次采样，发布流水线默认模型训练 1,000,000 步。本次的两组差值适用于该 checkpoint、索引和 20 对起终点。 [论文 pp.5–6、17–18](https://proceedings.mlr.press/v267/lee25f.html)

### 启动、续跑与产物

```bash
cd phase_b
run_dir=$(bash run_local.sh medium)
systemctl --user status "$(cat "$run_dir/unit.txt")"
tail -f "$run_dir/run.log"
# systemd 服务退出后，0 代表训练、评估和出图均成功。
cat "$run_dir/exit_code"
```

若本机作业被中断，从原目录恢复，定制训练参数须保持相同。

```bash
bash run_local.sh medium --resume "$run_dir"
```

训练、评估、出图在同一个 systemd 用户作业内顺序执行，共用本目录的 `uv` 环境和锁文件。休眠期间计算暂停；若重启导致作业终止，使用上面的续跑命令。整个作业在本机运行。

| 文件 | 内容 |
|---|---|
| `training_config.json`、`training_report.json`、`training.csv` | 数据哈希、超参数、环境记录、逐步 loss、计时和显存 |
| `training_state.pt`、`diffusion.pt` | 可续训状态、两组共用的采样模型，均保存在本机 |
| `medium_report.json`、`medium_crossings.csv` | 两种穿墙比例、配对计数、采样与索引资源逐项记录 |
| `medium_samples.npz` | 两组原始计划、逐线段命中标记、端点行号、种子和检索窗口编号 |
| `medium_metrics.png`、`medium_trajectories.png` | 训练 loss 与 artifact ratio 曲线、前四对起终点的配对轨迹 |
| `gpu.csv`、`run.log`、`unit.txt`、`exit_code` | 整卡采样、作业日志和退出状态 |

### 阶段 3–4 实测

2026-11-20 的本机运行完成 50,000 步，用时 **17 分 05 秒**，稳定均值 20.41 ms/步。首末各 250 步平均训练 loss 为 0.14976 和 0.05323。训练时 PyTorch 峰值分配约 292 MiB、保留 350 MiB，整卡采样峰值 514 MiB。CPU 建索引 13.62 秒，两组采样分别用时 5.99 秒和 8.05 秒。独立作业从启动到退出共约 **17 分 42 秒**，涵盖初始化、训练、建索引、评估和出图。

| 指标 | Diffuser | Diffuser + LoMAP | 变化 |
|---|---:|---:|---:|
| 10 次采样内至少一次穿墙的起终点对 | 12/20（60%） | 5/20（25%） | −35 个百分点 |
| 穿墙轨迹 | 43/200（21.5%） | 19/200（9.5%） | −12 个百分点 |

逐轨迹配对中，33 条从穿墙变为未穿墙，9 条发生反向变化，10 条两组都穿墙，148 条两组均未穿墙。轨迹图按预先固定的顺序展示前四对，第一对是局部退化的例子，由 1/10 增至 6/10。独立复算核对了 400 条计划的 102,000 个线段和所有计数。两组端点和 DDPM 随机流检查通过，开关组分别触发 270 和 0 次 Faiss 检索。另用一轮独立短训验证了从第 5 步恢复并重新生成未保存的步骤。

运行产物写入 `runs/`，其中可能包含机器环境信息、日志和模型权重；此目录已加入忽略规则，不应提交。复跑所需代码和依赖锁文件保存在本目录。
