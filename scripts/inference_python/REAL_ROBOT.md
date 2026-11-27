# 真机 / 仿真统一启动

在两个终端分别运行 `bash scripts/start_server.sh` 和 `bash scripts/start_client.sh`。
选择“运行环境”：`sim` 是原 ALOHA 仿真，`real` 是 Y1 SDK / CAN 真机。
切换环境会载入对应默认模型、B 样条 YAML、频率和输出目录，随后可继续修改。

## 真机服务端

默认配置为 `pi05_rtc_inference_dual_arm`，权重路径沿用原 `server_policy.sh`：
`checkpoints/pi05_base_full_dual_arm/bi_y1_test_20260410/30000`。
必须选择实际训练好的 Y1 权重，ALOHA 仿真 checkpoint 不适用于真机。
该配置支持同步、RTC 和 B 样条，新增元数据标明机器人、14 维动作、50 步窗口及 RTC 前缀空间。
服务端需要重启以提供这些元数据。单臂可使用 `pi05_base_full_single_arm` 配合匹配的权重，
支持同步 / B 样条；该配置不是 RTC 模型，客户端会拒绝 RTC 模式。

## 真机客户端

统一入口 `scripts/inference_python/run_policy.py`：

- `sync`：同步请求动作窗口，每 `interval` 步重新推理。
- `rtc`：单工作线程异步推理，发送观测时刻剩余的旧动作前缀，服务器按训练变换转换坐标与归一化。
- `bspline`：`WebsocketClientPolicy → ModelAdapter → AsyncBSplinePolicy → RealRobotEnv.step`。
  后台拟合融合，主线程按 Hz 取一个目标并通过 SDK 下发；不再叠加旧脚本的线性插值。

原同步和 RTC 脚本仍保留。新 RTC 入口复用仿真中验证的前缀对齐调度器，不使用旧真机脚本的 temporal 混合。
仿真调度器现支持指定动作维数，原仿真默认 14 维行为不变。

默认真机 B 样条配置：`packages/openpi-async_bspline/configs/y1_real.yaml`，30 Hz、8 步重规划。
调节控制频率时也要修改 B 样条 YAML 的 `fps`；不匹配会在初始化硬件前拒绝运行。
双臂旁路夹爪索引为 `[6,13]`；终端切换单臂时改为 `[6]`。
动作顺序是左臂六关节、左夹爪、右臂六关节、右夹爪；单臂默认右臂。
模型动作必须严格匹配 7 或 14 维，不会静默截取未知维度。

在客户端可编辑任务提示词、频率、相机类型、相机名称列表及硬件配置文件路径。
CAN 和设备映射在 `scripts/inference_python/configs/y1_hardware.yaml` 中修改。
V4L2 / Orbbec 现在按选择延迟导入，不再强制同时安装两种相机驱动。
真机每次启动执行一次有限步数任务；不会自动回到数据集初始位姿。

记录/绘图/录像菜单继续可用：

- 轨迹在预分配内存中记录，结束后写入 `OUTPUT/MODE/trajectory.npz`。
- `plots/` 保存结束后生成的曲线和指标。
- `camera.mp4` 保存第一路配置相机的视频。视频帧在内存保存至结束，容量由 steps 限制，
  大分辨率长时间录像有显著内存及复制开销；录制帧按控制频率编码，不代表精确采集时间。
- `bspline_diagnostics/` 保存已启用的样条诊断；`config.json` / `summary.json` 保存参数与完成步数。

SDK 下发的是位置目标，YAML 的速度/加速度输出用于诊断，不会转为驱动的速度控制命令。
代码未新增硬件急停/失能功能：退出停止下发并关闭相机，但现有 `RealRobotEnv.stop()` 不关闭电机。
SDK 初始化仍会使能机械臂。相机仍采用最新帧缓存，无严格多相机同步或帧龄校验。

## 无硬件验证

```bash
/tmp/openpi-bspline-venv/bin/python scripts/inference_python/run_policy.py \
  --mode bspline --mock --steps 60 --prompt test \
  --record-trajectories --plot-trajectories \
  --output /tmp/y1_bspline_smoke
```

`--mock` 不导入机器人/相机驱动、不连接模型服务器，运行同一套调度与记录流程。
切换为 `--mode sync` / `rtc` 可测试另两种模式，结果会标记 mock。
启动器的 `--dry-run` 仅生成命令和配置，不初始化硬件。

## 实机环境前提

在机械臂计算机指定 `OPENPI_CLIENT_PYTHON` 为装有硬件依赖的 Python，再启动客户端。
该环境需安装 `openpi-client`、`openpi-async_bspline`、`openpi-trajectory-recorder[plot]`、
终端 requirements，以及 Y1 SDK、OpenCV、rerun、所选 V4L2 或 pyorbbecsdk，录像另需 imageio/ffmpeg。
原 SDK 使用 `scripts/inference_python/assets/imeta_y1/y1_with_gripper.urdf`，需提供匹配的 URDF 和关联资源。

开发机当前没有 Y1 SDK、相机驱动、上述 URDF 和真机 checkpoint；已完成的检查是 mock/单元测试，
尚未在物理机械臂上验证。模型/消息元数据不替代机器人关节单位、夹爪量程与训练数据一致性的确认。
