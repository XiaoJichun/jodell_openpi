# 推理启动器配置

- `server.yaml`：模型服务器默认值，GPU、端口、模型配置名与 checkpoint。
- `y1_hardware.yaml`：CAN 接口和相机设备映射。
- `client.yaml`：客户端默认值，模式、地址、步数、prompt、相机、记录及输出等。

每份文件结构：

```yaml
default_target: sim  # 改为 real 可默认进入真机
common:             # 两个环境公共值
  ...
profiles:
  sim:              # 覆盖公共值
    ...
  real:
    ...
```

加载优先级：common → 对应 profiles → 本次交互编辑。
`--target` 覆盖 default_target。切换环境会重新加载该环境的默认值，丢弃尚未启动的菜单修改。
所有配置中的相对路径按项目根目录解析，不按 YAML 文件目录解析。
输出路径的 `{timestamp}` 在启动菜单时替换为 UTC 时间戳。

```bash
bash scripts/start_server.sh --target real
bash scripts/start_client.sh --target real
# 复制本文件夹 YAML 后，可指定自己的配置
bash scripts/start_client.sh --config /path/to/client.yaml --target real
```

例如只修改真机客户端，可在 client.yaml 的 profiles.real 下加：

```yaml
host: 192.168.1.20
steps: 600
prompt: Pick up the tape
video: true
```

GPU 编号应写字符串，如 `gpu: '0'`；布尔值写 true/false，不加引号。
加载器检查未知字段、缺失字段、重复键与类型，不会默默忽略拼写错误。

三层配置各司其职：

| 层 | 文件 | 用途 |
| --- | --- | --- |
| 启动配置 | server.yaml / client.yaml | 菜单默认值及启动命令 |
| 算法配置 | client 的 bspline_config 指向的 YAML | 拟合、融合、调度、内部诊断 |
| 硬件配置 | client 的 hardware_config 指向的 YAML | CAN、相机路径与序列号 |

B 样条模式使用算法 YAML 的 replan_interval，客户端 interval 只控制同步/RTC。
客户端 hz 和算法 fps 必须一致，不会隐式覆盖。
单臂模式启动菜单会将夹爪旁路改为 [6]；真实单臂服务端维数适配仍待完成。

菜单编辑不覆盖源 YAML。启动时（包括 dry-run）保存 launcher_时间戳.yaml，包含菜单最终值、
命令和环境变量；这是运行记录，不是可直接传入 --config 的 common/profiles 配置。
客户端快照位于输出目录，服务器快照位于 data/inference_launches/server。
B 样条另保存实际算法配置快照；真实执行程序仍保存自己的 config.json。

本目录不决定使用哪个 Python；启动前通过 `source .venv-*/bin/activate` 激活对应环境。
若设置了 OPENPI_SERVER_PYTHON / OPENPI_CLIENT_PYTHON，则这些变量优先覆盖已激活环境。
