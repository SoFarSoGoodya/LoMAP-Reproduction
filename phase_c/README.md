# LoMAP Phase C 容器运行

Phase C 用 D4RL Maze2D-Medium 和 Large 各训练一个 Diffuser。每个地图使用同一份 EMA checkpoint 做 Diffuser 与 LoMAP 的配对评估。正式配置每个地图从头训练 100 万步、batch 64、训练 seed `20261120`、评估 seed `20261123`，评估 20 对起终点、每对 30 条轨迹。CPU Faiss 从各自训练数据抽取 100 万个窗口，配置为 IVF-Flat、`nlist=1000`、`nprobe=5`。这些是本复现的评估规模与实现选择。 [论文 Figure 2](https://proceedings.mlr.press/v267/lee25f.html)

设 `LOMAP_ONLINE_GAP_WEIGHT=0.1` 可加入作者 LoMAP 仓库里的 Online restoration gap 分支。`0.1` 是试跑参数，原版 RGG 需要另外训练的 gap predictor。不开启时，主结果包含 Diffuser 与 LoMAP 两组。每一组都保存同样的起终点、DDPM 随机流和逐轨迹穿墙结果。穿墙条件为生成轨迹的任意线段进入地图墙体内部，Figure 2 的曲线统计随着每对采样数增加，至少有一条穿墙轨迹的起终点比例。 [作者 LoMAP 源码](https://github.com/leekwoon/lomap/tree/8ccb052481484352affaada26a0895102aef3e52) · [原版 RGG](https://github.com/leekwoon/rgg)

## 传输和预检

用平台控制台导入专用 SSH 公钥。将平台提供的 Host、用户名、端口配置为 `lomap-cloud`，首次连接前核对主机指纹。远程镜像需要能访问 NVIDIA GPU，具备 bash、git、tmux、tar、rsync、findmnt、flock、sha256sum 和运行 PyTorch/MuJoCo 的动态库。CUDA 先用实际算子与短训检验。 [CUDA 驱动要求](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html#cuda-driver)

```bash
ssh lomap-cloud 'findmnt --mountpoint /root/shared-nvme && mkdir -p /root/shared-nvme/lomap'
rsync -avP lomap-phase-c.tar lomap-phase-c.tar.sha256 lomap-cloud:/root/shared-nvme/lomap/
ssh lomap-cloud
cd /root/shared-nvme/lomap
sha256sum -c lomap-phase-c.tar.sha256
tar -xf lomap-phase-c.tar
cd lomap-phase-c
bash phase_c/preflight.sh
```

如镜像取依赖受限，用 `rsync -avP` 上传 `lomap-offline.tar.zst` 及其 `.sha256` 到同一持久盘目录。校验并用 `tar --zstd -xf lomap-offline.tar.zst` 解包，然后回到 `lomap-phase-c` 目录执行 `LOMAP_OFFLINE_ENV=/root/shared-nvme/lomap/lomap-offline bash phase_c/preflight.sh`。离线包为 Linux x86_64、Python 3.13，依赖由随包的 uv 按锁文件安装。保持解包目录路径可用，虚拟环境引用其中的 Python。使用同一容器和软件版本续训可以满足训练脚本的配置校验。

## 短跑与正式作业

在预检通过后执行以下命令。作业和状态均写在 `/root/shared-nvme/lomap/runs/<运行名>/`。

```bash
tmux new-session -d -s lomap 'bash phase_c/run.sh smoke medium cloud-smoke-medium'
tail -f /root/shared-nvme/lomap/runs/cloud-smoke-medium/job.log
```

给 `training.pid` 中的训练进程发 `SIGTERM`，确认退出状态是 75，再用相同参数与目录重启。应在日志看到 `Resuming from step`，最终状态为 `complete 0`，并通过 `sha256sum -c SHA256SUMS`。Large 用 `smoke large cloud-smoke-large` 独立验证。若正式运行打算开启在线 gap，短跑时也开启并保持配置一致。

根据两张地图实际短跑的稳定吞吐安排正式任务。优先 Large，通过预检的 Medium 可在 Large 阻塞时先开跑。只有在估算时限覆盖两张图及评估时才串行排两项。例如两张图都已通过并且窗口充足时，可在同一 tmux session 用 `&&` 顺序执行。

```bash
tmux new-session -d -s lomap 'bash phase_c/run.sh formal large c-large-seed20261120 && bash phase_c/run.sh formal medium c-medium-seed20261120'
```

断开 SSH 后重连，确认正式作业步数持续增长且第一份 checkpoint 可读。验收时核对 `status.tsv`、`training_report.json`、逐轨迹 CSV、原始 NPZ、两张 PNG 和 `SHA256SUMS`。训练退出后在平台控制台停算力并保留数据盘，按平台支持的方式下载和核对哈希。
