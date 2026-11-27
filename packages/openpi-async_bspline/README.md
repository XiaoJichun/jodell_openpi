# openpi 异步 B 样条插件

独立、可选的客户端插件，无需安装 ominf，也不修改模型、服务端或原有同步客户端。
适用于服务端返回 `{"actions": ndarray[H, D]}` 的位置动作策略。

## 安装

在 openpi 仓库根目录使用 Python 3.11 或更新版本：

```bash
pip install -e ./packages/openpi-client -e ./packages/openpi-async_bspline
```

只需要客户端、NumPy、SciPy，不需要安装训练环境或 GPU 依赖。
插件独立于根目录的 uv 训练工作区；也可用 `uv pip install -e ./packages/openpi-client -e ./packages/openpi-async_bspline` 安装到现有环境。

## 接入

推荐使用 YAML 配置。ALOHA 模板为 [`configs/aloha_sim.yaml`](configs/aloha_sim.yaml)：

```yaml
bspline:
  degree: 3
  max_deriv: 2
  knots_interior: 6
  knots_interior_merge: 3
  merge_input_mode: crop
  replan_interval: 15
  blend_lead_steps: 3
  passthrough_indices: [6, 13]
```

```python
policy = AsyncBSplinePolicy(client, config="packages/openpi-async_bspline/configs/aloha_sim.yaml")
```

文件仅在构造时读取一次，不在推理循环中解析或重读。也可以使用
`BSplineConfig.from_yaml(path)` 加载后传入，或继续使用下方 Python 参数形式。
YAML 支持扁平参数、`bspline:` 段以及 ominf 的 `env.bspline:` 层级。
有多个同名配置段时必须通过 `BSplineConfig.from_yaml(path, section="env.bspline")` 指定。
未知字段、重复 YAML 键、错误类型和不合法的次数/导数阶数会在启动前报错。

与 ominf 参数的关系：

| 参数 | 插件行为 |
| --- | --- |
| `degree` | 初始拟合和融合次数；当前支持 `max_deriv + 1 <= degree <= 5` |
| `max_deriv` | 融合内部约束阶数，至少为 1；仍不承诺激活旧/新轨迹时速度、加速度连续 |
| `knots_interior` | 新窗口拟合内部节点数，按窗口样本数限制上限 |
| `knots_interior_merge` | `refit` 模式旧段重新拟合内部节点数；`crop` 不重新拟合，故不使用此值 |
| `merge_input_mode` | `crop` 裁剪缓存拟合；`refit` 切散点后重新拟合左右两段 |
| `replan_interval` / `blend_lead_steps` | 当前插件调度参数，以控制步计数；blend_lead_steps 为 offset_timestep 未设置时的兼容选项 |
| `passthrough_indices` | 跳过平滑的维度，例如 ALOHA 夹爪 6、13 |

原始 `ominf/src/ominf/trajectory` 读取的配置定义在 `ominf/configs/*/*.yaml` 的 `env.bspline`。
现在模板列出了原 `BsplineConfig` 的 **全部 24 个字段**。适配行为如下：

| 参数 | 插件行为 |
| --- | --- |
| `interpolate_factor` | 工作线程按 `ceil(horizon × factor)` 建立均匀插值缓存，再生成控制步数组；0 表示精确求值 |
| `offset_timestep` | 推理返回时的当前控制步加此偏移作为拼接点，允许小数；与 `blend_lead_steps` 二选一 |
| `publish_position/velocity/acceleration` | 返回 `positions/velocities/accelerations`；OpenPI 的 `actions` 始终保留 |
| `bspline_infer_plot` | 在内存保留拟合窗口，结束后输出各阶段位置、速度、加速度图 |
| `bspline_publish_plot` | 预分配内存记录实际发布位置，结束后差分分析并绘图 |
| `timing_print` | 记录工作线程推理加平滑耗时，保存时输出汇总；不在控制循环打印 |
| `plots_dir` | `save_diagnostics()` 默认输出目录 |
| `joint_col` / `plot_ncol` | 本插件将 joint_col 适配为逗号分隔关节名称；plot_ncol 控制图列数 |
| `plot_joint_indices/point_indices/merge_steps` | 逗号分隔的关节索引、原始窗口采样点索引、窗口编号；前两者从 0 开始，窗口编号从 1 开始且包括初始拟合 |
| `plot_raw/scatter/c2_merge/final` | 显示原始样本、样本拟合曲线、必经点修正前融合曲线、最终曲线 |
| `plot_overall` | 配合 publish plot 输出每个 episode 所有关节的总览 PDF |

原始文件可直接选取 `env.bspline`，并继承 `env.fps`。这些是 OpenPI 适配语义，
不是原机器人发布接口的完整复制。`fps` 用于导数单位换算，不负责控制循环定时。
平滑维度使用解析导数；旁路夹爪和实际执行记录使用有限差分。
合并阶段诊断图仅显示拼接点之后的有效区间，执行全程请看 publish 图或独立 recorder。

Python 默认保持 `interpolate_factor=0`、速度/加速度输出关闭；完整 YAML 模板显式启用
3 倍缓存及速度/加速度输出。`offset_timestep=0` 没有预留工作线程计算余量，可能触发拼接超时。
`diagnostic_capacity` 和 `publish_capacity` 限制记录容量，超出后丢弃新增记录并计数。
这些诊断有少量内存复制成本；绘图、压缩和文件写入全部推迟到显式保存。

```python
with AsyncBSplinePolicy(client, config="packages/openpi-async_bspline/configs/aloha_sim.yaml") as policy:
    # 执行控制循环；在 YAML 中打开需要的 bspline_*_plot 开关
    ...
policy.save_diagnostics()  # 必须在 close() 后；输出 NPZ、PNG、PDF、配置和耗时 CSV
```

绘图需要安装 `pip install -e 'packages/openpi-async_bspline[plot]'`。
示例 demo 和 benchmark 在结束并关闭策略后自动保存已启用的诊断。
跨同步 / RTC / B-spline 的执行轨迹对比仍使用独立的
[`openpi-trajectory-recorder`](../openpi-trajectory-recorder/README.md)。

通过 YAML 运行仿真：

```bash
python examples/aloha_sim/benchmark.py --mode bspline \
  --bspline-config packages/openpi-async_bspline/configs/aloha_sim.yaml \
  --record-trajectories --episodes 10 --output data/aloha_sim/yaml_run
```

使用 YAML 时请在文件内设置调度参数，不再同时指定 `--interval` / `--blend-lead`。
`config.json` 会记录配置路径和最终生效的 `bspline_settings`。
无模型的二维演示使用 `python examples/bspline_demo.py --config packages/openpi-async_bspline/configs/demo.yaml`。

用 `AsyncBSplinePolicy` 替换 `ActionChunkBroker`，不要将二者嵌套：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi_async_bspline import AsyncBSplinePolicy, BSplineConfig

client = WebsocketClientPolicy(host="localhost", port=8000)
with AsyncBSplinePolicy(
    client,
    BSplineConfig(
        replan_interval=15,
        blend_lead_steps=3,
        knots_interior=6,
        passthrough_indices=(),  # 根据动作定义填写夹爪等不应平滑的列
    ),
) as policy:
    # 在现有固定频率控制循环中：
    observation = get_observation()  # 替换成你的环境读观测函数
    action = policy.infer(observation)["actions"]  # 一步动作，shape=(D,)
    send_action(action)  # 替换成你的控制函数
```

也可直接传入现有 `PolicyAgent(policy)`。由原控制循环定时调用 `infer()`；插件不启动机器人控制循环。
每次调用消耗一个逻辑时间步，预测动作间隔应与控制间隔一致。本版本不做不同频率间的重采样。
首窗会等待推理和拟合；之后推理、拟合、融合在一个后台线程串行进行，控制线程读取预采样数组。
观测会在提交时复制；图像复制和 Python 线程调度意味着这不是硬实时系统。

默认参数用于 H=50 的动作块。H 必须大于 `replan_interval + blend_lead_steps + 1`。
后台请求的第 0 个动作对应提交观测时的逻辑步；推理返回后选择当前播放步之后
`blend_lead_steps` 的位置作为必过点，保留旧轨迹前缀并融合新轨迹。
推理延迟和拟合耗时必须留在两窗重叠范围内；应依据实际控制频率和延迟调整参数。
`knots_interior` 越小通常平滑越强，短窗口会自动限制节点数量。

`passthrough_indices` 中的列沿用旧窗口前缀、随后直接使用新窗口的离散原值。
它们不承诺连续性。默认平滑全部列；旋转表示、相对动作、速度命令需要调用方先明确语义，
不能直接视为关节位置轨迹。
返回值保留最近激活窗口的其它元数据，不会按步切片其它数组。

## 错误与生命周期

- 无重叠、融合失败、计算错过切换时刻或旧轨迹耗尽时抛异常；不静默回放最后动作或切换原始轨迹。
  控制端应捕获异常并执行自身的停止逻辑，然后 `reset()` 开启新回合。
- `reset()` 等待在途推理结束后再重置底层策略，旧结果不会进入新回合。
- `close()` / `with` 退出会等待后台线程结束。底层网络应配置可终止的超时；
  Python 无法强制取消已经运行的推理，若底层永久阻塞，reset/close 也会等待。
- 从同一控制线程调用 infer/reset/close。被包装的策略不应再由其它调用方并发使用。
  关闭插件不会关闭调用方拥有的传输连接。

## 来源与算法边界

`bspline_c2_merge.py` 迁自 `ominf/src/ominf/trajectory/bspline_c2_merge.py`，
保留三次最小二乘拟合、crop、C² 样条融合及必过点修正；修复原
`merge_c2_with_must_pass` 中局部变量遮蔽函数的问题。
`trajectory.py` 将原 `trajectory_buffer.py` 的窗口拟合、必过点、Stitch 思路适配成
不可变窗口，去除了 ominf 配置、消息类型、日志、画图和机器人依赖。
后台调度由 `policy.py` 实现，不依赖 ominf 的异步服务协议。

融合样条内部的 C² 连续性，不等同于切换旧轨迹与新样条时的速度、加速度连续。
原必过点修正仅约束位置；本插件延续此行为。最小二乘夹紧节点也不保证首尾样本严格插值。
该算法不提供关节限位、速度/加速度限幅或碰撞检测。

## 可选曲线对比插件

安装绘图扩展（仅需 CPU，不加载模型）：

```bash
pip install -e ./packages/openpi-client -e './packages/openpi-async_bspline[plot]'
openpi-trajectory-plot data/aloha_sim/a100_results_20261125 --output data/aloha_sim/trajectory_plots
```

打开输出目录中的 `index.html`，选择种子、动作指令/观测数据、位置/速度/加速度。
每张图覆盖全部 14 个维度，左列为左臂，右列为右臂；三种策略使用固定颜色叠加。
默认绘制所有种子，也可通过 `--seeds 0 1` 选择回合，`--dpi 180` 提高 PNG 分辨率。
输出目录必须不存在，防止覆盖已有报告。

输入为 `sync/`、`rtc/`、`bspline/` 下的 `seed_N.npz`（`actions` 和 `states` 均为 `(N, 14)`），
以及记录频率的 `config.json`（`args.hz`）。支持 `--modes` 指定其他模式目录名，
缺失配对回合会报错。自定义数据可使用 `--hz 50` 显式指定采样频率。
此绘图模块目前使用 ALOHA 的 14 维关节定义，其他机器人需先适配维度名称和单位。

输出包括：

- 每回合 6 张 PNG：指令和观测各包含位置、速度、加速度对比。
- 每回合 2 份三页 PDF：便于缩放和导出。
- `metrics.csv`：按回合、模式、关节统计各曲线 RMS 和绝对峰值，仅使用三模式共有的时间前缀。
- 每回合 `actions_MODE.npz` / `states_MODE.npz`：保存 `position_time/values`、
  `velocity_time/values`、`acceleration_time/values`，便于二次分析。
- `manifest.json`：采样频率、轨迹长度、时间轴和计算方法说明。

速度使用一阶差分并放在采样中点，加速度使用二阶差分并放在内部采样时刻。
不做滤波、端点外推或任务阶段对齐；不同长度的曲线在各自回合结束时停止。
横轴为名义仿真时间，不是实际墙钟时间。`states` 是施加当前动作之前记录的位置，
其速度/加速度也是差分估计，并非直接测量值。第 6、13 列夹爪使用归一化开合量，其他关节单位为弧度。

Python 接口：

```python
from openpi_async_bspline.plotting import build_report

build_report("data/aloha_sim/a100_results_20261125", "data/aloha_sim/selected_plots", seeds=[0])
```

本机已生成全部 10 回合的报告：`data/aloha_sim/a100_results_20261125/plots/index.html`。

## 验证

```bash
pip install pytest
pytest packages/openpi-async_bspline/tests
python examples/bspline_demo.py
```

演示使用本地模拟策略，无机器人、模型服务器依赖。
