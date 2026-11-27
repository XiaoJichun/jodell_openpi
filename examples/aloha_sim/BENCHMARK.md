# ALOHA Sim：同步 / RTC / 异步 B 样条对比

使用公开的 `pi0_aloha_sim` **JAX 权重**测试方块移交任务，不需要下载训练数据或重新训练。
`benchmark.py` 提供 `sync`、`rtc`、`bspline` 三种模式。后者使用 `openpi-async_bspline` 插件。

## 交互式启动（本机现有环境）

在两个终端中分别执行：

```bash
# 终端 1：服务端
bash /root/workspace/bspline_test/openpi/scripts/start_server.sh

# 终端 2：等服务端开始监听后，启动客户端
bash /root/workspace/bspline_test/openpi/scripts/start_client.sh
```

终端采用与 `ominf/toolkits/data_process` 相同的 **questionary 方向键菜单**：

- **↑ / ↓** 移动，**Enter** 选择或提交。
- **Space** 勾选/取消多选项。
- **Ctrl+C** 取消当前输入或返回；在主菜单退出。
- 文本框预填当前值，可直接编辑；路径输入支持 Tab 补全。

服务端可选择 GPU、端口、模型配置、checkpoint 和 JAX 显存比例，最后选择“启动服务端”。
三模式对比使用默认 `pi0_aloha_sim_rtc` 服务配置。
客户端选择“模式”，进入 **同步 / RTC / B 样条** 单选菜单；
选择“记录 / 绘图 / 录像”，用空格勾选需要的功能。绘图开启时自动启用记录。
其余菜单项可修改地址、端口、回合数、步数、种子、请求间隔、RTC 引导、延迟、mock、
渲染后端和输出目录。ALOHA 仿真固定 50 Hz。

选择“B 样条参数设置”进入分组菜单：

- 拟合与融合：次数、节点、融合模式、插值密度。
- 异步调度与输出：请求间隔、拼接偏移、fps、旁路维度和导数输出。
- 诊断记录与容量：内部记录开关、输出目录、内存容量。
- 绘图内容与筛选：关节、采样点、窗口、曲线阶段和布局。

布尔值使用 Y/N 输入，枚举使用单选菜单，其余参数使用预填文本框。
索引列表输入 `[6,13]`，逗号分隔筛选字符串输入 `0,4`；清空文本框可保存空字符串。
修改后选择“校验并应用参数”，或“放弃本次修改并返回”。
客户端还提供“加载 YAML 配置”，可以加载其他配置文件。
B 样条模式的请求间隔由 YAML 的 `replan_interval` 控制。

启动 B 样条客户端时，会在输出目录保存 `bspline_config_<时间>.yaml`，原配置模板保持不变。
本次 `plots_dir` 自动设为输出目录下的 `bspline_diagnostics`，避免不同运行混用诊断文件。
执行记录默认开启，回合结束后保存和绘图；内部样条诊断需在“B 样条参数设置”中打开相应开关。
默认输出目录带 UTC 时间戳。已有相同模式结果时会要求更换输出目录。

当前两个 Python 环境已安装终端组件；新环境需额外安装：

```bash
/path/to/python -m pip install -r scripts/interactive/requirements.txt
```

`--plain` 可切回编号菜单，非 TTY（例如管道输入）自动使用编号菜单。
两个脚本都支持 `--dry-run`：仍可交互修改参数，最后显示启动命令而不运行推理；
B 样条客户端仍保存配置快照。脚本可从任意工作目录启动。
推理启动后进程占用当前终端，可用 Ctrl+C 停止；修改参数需重新运行脚本。

默认使用 `/tmp/openpi-server-venv/bin/python` 和 `/tmp/openpi-bspline-venv/bin/python`。
如环境位置变化，可显式指定已安装相应依赖的 Python：

```bash
OPENPI_SERVER_PYTHON=/path/to/server/bin/python bash scripts/start_server.sh
OPENPI_CLIENT_PYTHON=/path/to/client/bin/python bash scripts/start_client.sh
```

真机接入与模式切换见 [真机部署说明](../../scripts/inference_python/REAL_ROBOT.md)。两个启动器均可通过“运行环境”切换真机 / 仿真。

## 安装客户端

在仓库根目录创建独立的 Python 3.11 环境，避免混入训练依赖：

```bash
uv venv --python 3.11 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip install -e ./packages/openpi-client -e ./packages/openpi-async_bspline \
  gym-aloha==0.1.1 mujoco==2.3.7 dm-control==1.0.14 imageio imageio-ffmpeg
```

渲染需要可用的 EGL；无 GPU 的客户端可以安装系统 OSMesa 库并设置 `MUJOCO_GL=osmesa`。
模型服务端可在另一台有 GPU 的机器上，客户端通过 `--host`、`--port` 连接。

## 启动同一个权重的服务端

在配置好原 openpi / JAX 环境的另一终端执行（首次启动会下载模型）：

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_aloha_sim_rtc \
  --policy.dir gs://openpi-assets/checkpoints/pi0_aloha_sim
```

这个配置沿用官方 `pi0_aloha_sim` 的模型尺寸、动作定义、`Transfer cube` 提示词和归一化资产，
配置参考 [官方 ALOHA Sim 配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)。
只将采样实现换成仓库已有的 `Pi0RTC`。三种评测模式都使用该服务端：
同步和 B 样条请求设置 `enable_rtc=False`，RTC 设置为 True。
因此比较的是相同权重、相同采样实现中不同的调度/引导方式，不混用不同任务模型。

也恢复了普通 `pi0_aloha_sim` 配置，供原 `examples/aloha_sim/main.py` 使用；
本文的三组对比统一要求 RTC 服务端，避免基线和 RTC 的服务配置混淆。

## 运行对比

在客户端终端依次执行，三组默认都是种子 0–9、50 Hz、每 15 步发起一次新请求：

```bash
MUJOCO_GL=egl python examples/aloha_sim/benchmark.py --mode sync --episodes 10
MUJOCO_GL=egl python examples/aloha_sim/benchmark.py --mode rtc --episodes 10
MUJOCO_GL=egl python examples/aloha_sim/benchmark.py --mode bspline --episodes 10
```

每组运行前建议以相同命令重启服务端，使模型 RNG 初始状态一致。客户端固定环境种子；
异步模式的调度和请求数仍会随机器耗时变化，因此不是逐动作相同噪声的配对实验。
所有模式先预热两次；RTC 的第二次预热包含前缀以触发对应 JIT 编译。
每回合第一段动作的等待单独记为 `bootstrap_ms`，不混入持续控制周期统计。

- `--delay-ms 100`：在每次推理前额外增加 100 ms 延迟，用于压力测试；三组应使用同一数值。
- `--output data/aloha_sim/delay100`：不同实验使用不同输出目录，已有模式目录不会被覆盖。
- `--video`：保存录像；渲染/录像开销会影响控制周期，比较时三组保持一致。
- `--steps 400`：回合上限；任务奖励达到 4 表示成功，失败/超时计入总回合数。
- `--blend-lead 3`：B 样条未来切换点的预留步数；夹爪维度 6、13 不做样条平滑。
- `--guidance 0.5`：RTC 最大引导权重。

RTC 前缀使用**当前观测时刻尚未执行的旧动作**。新响应按实际已执行步数丢弃过期部分。
客户端发送 `rtc_prefix_space="environment"`，服务端对前缀执行与训练相同的
ALOHA 动作坐标转换、归一化、维度补齐，再交给 RTC；未声明该字段的旧客户端行为不变。

## 输出与解释

可选运行时记录插件见 [`openpi-trajectory-recorder`](../../packages/openpi-trajectory-recorder/README.md)。
安装后加 `--record-trajectories`，用预分配内存记录动作/状态，回合结束并等待在途推理后保存
`MODE/recordings/seed_N.npz`。`--plot-trajectories` 额外在回合结束后生成 PNG/PDF/CSV，
绘图不发生在控制循环内。插件独立于 B 样条，可以同样包装 sync 和 RTC。

结果保存到 `data/aloha_sim/benchmark/{sync,rtc,bspline}/`：

- `config.json`：参数与服务端元数据。
- `summary.json`：成功率、错误回合数及每回合统计。
- `seed_N.json`：奖励、耗时、RPC 次数及延迟分位数、动作变化指标。
- `seed_N.npz`：逐步动作、观测状态、调用耗时和控制周期工作耗时。
- `seed_N.mp4`：启用 `--video` 时生成。

`control_overrun_fraction` 统计调用策略加仿真推进/渲染的工作时间是否超过 20 ms；
睡眠不计入工作时间。`wall_seconds_excluding_bootstrap` 记录实际墙钟时间。
命令速度、加速度、jerk 由关节目标的差分按 **50 Hz 名义物理时间**计算，
排除了夹爪，不是关节传感器直接测得的速度/力矩。`command_joint_step_max_rad`
是整回合最大相邻命令变化，包含接缝也包含正常运动，不是单独的接缝指标。
RPC 指标包含首窗和结束时等待完成的在途请求，但不含回合前 JIT 预热。

**本版本物理仿真在阻塞推理期间暂停。** 成功率用于固定仿真步长下的任务质量对比；
墙钟耗时和超时率用于延迟对比。它不模拟真实机器人在等待网络时继续运动，
不能仅凭成功率得出实时控制优劣。过大延迟导致 RTC/B 样条轨迹耗尽时明确记录错误，
不会静默保持最后动作来隐藏失败。
B 样条沿用原算法的位置必过点约束，不能保证新旧轨迹切换处的速度/加速度连续。

## 无模型冒烟测试

```bash
MUJOCO_GL=egl python examples/aloha_sim/benchmark.py \
  --mode bspline --mock --episodes 1 --steps 80 --output data/aloha_sim/smoke
```

`--mock` 仅保持观测关节位置，RTC 模式也不会真的执行神经网络引导。
所有输出会标记 mock，不应当作模型效果或任务基准结果。

不依赖 GPU、模型、MuJoCo 渲染的单元测试：

```bash
uv pip install pytest
PYTHONPATH=src python -m pytest --noconftest \
  examples/aloha_sim/benchmark_test.py \
  src/openpi/policies/rtc_inputs_test.py \
  packages/openpi-async_bspline/tests
```

`--noconftest` 用于跳过根项目测试套件的 NVML 初始化，这些测试只需要客户端和 NumPy/SciPy。

## 本次验证范围

- 28 项单元测试通过：包含动作前缀转换、RTC 时序对齐、B 样条多窗口、错误处理和指标记录。
- 2026-11-25 已在 A100-SXM4-80GB 上恢复真实 checkpoint，完成 JAX 普通推理与 RTC 引导编译。
- 使用 EGL 渲染，三个模式各通过 60 步真实模型短回合，再各执行 10 个正式回合，无运行错误。
- 相同种子 0–9、名义 50 Hz、重规划间隔 15 步、RTC guidance=0.5、B 样条 blend-lead=3，无额外延迟。
- 每种模式重启服务，均保存视频。请求上限 400 步，但当前模拟器会在 300 步截断，成功时提前结束。
- 修正了 RTC 首次无前缀请求与预热参数不一致引发的额外 JIT 编译，并重新执行 RTC 十回合；首请求耗时由约 2.35 s 降至约 98 ms。

最终结果位于 `data/aloha_sim/a100_results_20261125/`，包括 `comparison.json`、`comparison.csv`、
三种模式的逐回合视频/轨迹/指标、服务端和客户端日志，以及两套环境的依赖版本记录。

| 模式 | 成功回合 | 控制调用 P95 (ms) | 完整周期超 20 ms | 回合 jerk RMS 中位数 (rad/s³) |
| --- | ---: | ---: | ---: | ---: |
| 同步 | 6/10 | 100.129 | 16.2% | 515.21 |
| RTC | 4/10 | 0.110 | 16.8% | 892.56 |
| 异步 B 样条 | 5/10 | 0.080 | 43.7% | 184.96 |

控制调用 P95 汇总所有回合的非首窗调用；完整周期包括仿真推进和渲染。
后台模型请求的逐回合 P95 中位数分别为 104.2、156.9、113.1 ms，不能把亚毫秒控制调用理解为模型推理耗时。
B 样条 jerk 中位数比同步低约 64%，但完整周期超时较多，尚未实现稳定 50 Hz。
不同回合的轨迹长度和任务结果不同，仅十个种子也不足以确认成功率上的统计优势。
异步调度受实际墙钟耗时影响，即使初始种子一致，重跑也可能得到不同结果。

本机已准备的服务端环境为 `/tmp/openpi-server-venv`，客户端为 `/tmp/openpi-bspline-venv`。
使用现有环境启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OPENPI_DATA_HOME="$PWD/.cache/openpi" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 \
  /tmp/openpi-server-venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_aloha_sim_rtc --policy.dir ./checkpoints/pi0_aloha_sim
```

客户端使用 `MUJOCO_GL=egl /tmp/openpi-bspline-venv/bin/python examples/aloha_sim/benchmark.py ...`。
模型服务需在可访问 GPU 的执行环境中启动；默认隔离环境看不到 GPU。
这些 `/tmp` 环境可能被系统清理，依赖版本已保存在结果目录的 `*_requirements.txt` 中。

本地 checkpoint 已下载到 `checkpoints/pi0_aloha_sim`（约 11.19 GiB，20 个文件均通过官方 MD5 校验）。
启动时可将上述 `--policy.dir` 换成 `./checkpoints/pi0_aloha_sim`，无需重新下载。
下载来源及校验记录保存在 `checkpoints/pi0_aloha_sim.download_receipt.json`。
