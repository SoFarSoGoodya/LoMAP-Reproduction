# LoMAP 论文复现

本项目在本机验证 LoMAP 的轨迹投影机制，并在 D4RL Maze2D-Medium 上做 Figure 2 左图的小规模穿墙对照。 [LoMAP 论文](https://proceedings.mlr.press/v267/lee25f.html)

| 实验 | 内容 | 已完成 |
|---|---|---|
| [A · 二维合成轨迹](phase_a/README.md) | CPU 单步投影，检查避障和轨迹误差 | 主配置碰撞数 110/256 → 46/256 |
| [B · D4RL Maze2D](phase_b/README.md) | U-Maze 试跑、Medium 短训及配对评估 | Medium 训练 50,000 步。每对采样 10 条，出现穿墙的起终点对 12/20 → 5/20 |

两个目录各自包含运行说明、代码和 `uv.lock`。A 的结果文件保存在 `phase_a/results/`；B 的训练和评估产物默认写入本机 `phase_b/runs/`，其中含模型权重和运行环境信息，不纳入版本控制。B 首次运行前需要下载固定版本的上游源码与 Maze2D 数据，并生成 MuJoCo XML 资产；后台运行入口使用 Linux systemd 用户服务，并支持从 checkpoint 续训。许可证范围和第三方材料见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 待做

- [ ] 增加训练 seed 和起终点样本，检查 Medium 穿墙率下降的稳定性。
- [ ] 扩展 Figure 2 对照到每对最多 30 条轨迹，补齐 RGG baseline 和 Maze2D-Large。
- [ ] 按发布配置做百万步训练，评估训练预算和检索库规模对结果的影响。
