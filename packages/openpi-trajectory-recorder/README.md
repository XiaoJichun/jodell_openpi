# openpi-trajectory-recorder

独立的运行时内存记录插件。仅依赖 NumPy 和 openpi-client，不依赖 B 样条、模型、GPU 或网页。
支持任意维度的逐步位置策略；Matplotlib 仅在执行后的分析阶段导入。

## 安装

```bash
pip install -e ./packages/openpi-client -e './packages/openpi-trajectory-recorder[plot]'
```

## 插入已有控制循环

```python
from openpi_trajectory_recorder import RecordedPolicy

# policy 是最终逐步输出的策略：同步 broker、RTC 或 AsyncBSplinePolicy 均可。
# 必须放在 chunk 调度器外层，不能直接包装返回 (H,D) 的原始模型 RPC。
recorded = RecordedPolicy(policy, capacity=30_000, action_dim=14, hz=50)
recorded.reset()
for _ in range(30_000):
    obs = env.get_observation()
    result = recorded.infer(obs)
    env.apply_action(result)
    if env.is_episode_complete():
        break

# 控制循环和在途推理结束后执行：
# 若 policy 有后台工作，使用其 reset()/close() 等待完成。
recorded.reset()
path = recorded.save("data/my_run/sync/seed_0.npz", metadata={"mode": "sync", "seed": 0})

# 需要时绘图，输出 PNG、PDF、CSV，没有网页。
from openpi_trajectory_recorder.analysis import analyze

analyze([path], "data/my_run/sync/figures", labels=["sync"])
```

记录前预分配并触碰全部内存页；每步只复制动作/状态向量并记录时间，不复制图像，
不计算速度/加速度，不创建工作线程，不做磁盘 I/O，不打印日志，不绘图。
输出对象原样返回；策略异常原样传播。包装器只记录返回的指令，不能证明执行器已实际执行。

严格“零开销”无法保证。NumPy 向量的复制和计时有微小 CPU 开销，应在目标控制机测量。
14 维动作/状态、30,000 步，预分配约 6.87 MiB；运行时不扩容。
记录满后保留已有数据、跳过新增数据，并增加 `recorder.dropped`，不会阻塞或覆盖旧数据。
形状错误记录到 `invalid`/`last_error`，不改变策略返回值；异常计入 `failed_inferences`。
含缺失/错误样本的记录可保存，但分析器拒绝用均匀时间差分生成误导性曲线。

`reset()` 仅转发底层策略，不清空数据。每回合新建一个记录器，或保存后显式 `recorder.clear()`。
所有记录、保存、清空操作由同一控制线程执行，不支持并发读写。插件不拥有底层策略，不隐式关闭它。
`save()` 必须在控制结束后调用，拒绝覆盖文件；绘图输出目录也必须不存在。

如果不需要策略包装，也可直接调用 `TrajectoryRecorder.append(actions, states, timestamp=...)`，
在发送控制命令的位置插入记录。此时调用方负责传入正确的动作前状态和采样时刻。
`state_dim` 可与 `action_dim` 不同；包装器可配置 `state_key`、`action_key`。

## 推理后比较三组数据

```bash
openpi-trajectory-analyze \
  data/my_run/sync/seed_0.npz data/my_run/rtc/seed_0.npz data/my_run/bspline/seed_0.npz \
  --labels sync rtc bspline --output data/my_run/comparison
```

每组包含位置、速度、加速度；动作指令和观测状态分别绘制。
输出 6 张 PNG、6 页 PDF、逐维度 RMS/峰值 CSV 和数据说明 JSON。
记录文件包含动作、动作前状态、逐次推理耗时、墙钟相对时间、名义频率及丢样计数。
速度/加速度在执行后由未滤波位置差分计算，不是直接测得的关节速度/加速度。
图的横轴使用名义采样时间，不外推回合尾部；统计仅使用共有时间段。
14 维默认采用 ALOHA 定义：夹爪 6/13 为归一化开合量，其余为弧度。其他维度标为通用位置单位。
旧版 benchmark NPZ 也可分析，需要加 `--hz 50`。

## 已接入 ALOHA benchmark

```bash
python examples/aloha_sim/benchmark.py --mode bspline --episodes 10 \
  --record-trajectories --output data/aloha_sim/recorded_run
```

记录在每回合结束并等待在途请求后保存到 `bspline/recordings/seed_N.npz`。
如需每回合结束后立即绘图，改用 `--plot-trajectories`（同时启用记录）。
sync、rtc 使用相同开关。视频采集是独立开关，记录器自身不复制图像。

## 本机验证与开销

- 39 项插件与调度相关测试通过，包含输出对象保持、观测快照、溢出、异常、无运行时 I/O 和导数校验。
- MuJoCo EGL + B 样条保位 mock 实测 80 步，无丢样；保存的指令和状态与 benchmark 记录一致。
- 既有真实轨迹的三模式回放验证通过，图表保存在 `data/aloha_sim/recorder_replay_20261125/comparison/`。
  此项是已有数据回放，不是新的神经网络推理评测。
- 14 维 NumPy CPU 微基准，每批 50,000 次，8 批、丢弃首批后取中位数：
  额外开销约 **2.93 µs/步**，约占 20 ms 周期的 **0.015%**。这不是 GPU 端到端或硬实时保证。

可在目标机器复测：

```bash
python examples/trajectory_recorder_overhead.py --output data/recorder_overhead.json
```
