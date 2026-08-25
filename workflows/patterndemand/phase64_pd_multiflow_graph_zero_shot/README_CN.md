# Phase64：PD 多流通信图零样本验证

## 这一步回答什么

Phase51 已有 P1D1 单链路 L1/L2/L3 物理曲线；Phase60–63 已证明冻结的 R61 轻量修正对 P1D2/P2D1 两条并发流可迁移到六个模型。Phase64 不重新训练，而是检查同一规律能否按“谁同时发给谁”的通信图，扩展到最多四条流：

- `P1D4`：一个 P 同时发给四个 D，验证四路 fan-out；
- `P4D1`：四个 P 同时发给一个 D，验证四路 fan-in；
- `P2D2_MATCHING`：两条互不共享端点的一对一并发，检查全局 fabric 干扰；
- `P2D2_ALL_TO_ALL`：四条流同时共享两侧端点，检查最忙端点的带宽争用。

代表模型是 `qwen3-8b`（常规 KV、page1）与 `deepseek-v2-lite`（MLA、page64）。每个配置测 L1/L2/L3、两个预冻结 placement、10 个冻结 payload vector，共 48 个 shard、240 个 official point。

## 冻结预测公式

每条边先从 Phase51 曲线取得单链路代价 `C_e`；令 `M=max(C_e)`，`B=max(每个P的出边代价和, 每个D的入边代价和)`：

```text
T_hat = max(1, intercept + (beta_max-beta_min)*M + beta_min*B)
```

三个系数来自 R61，不允许用 Phase64 标签调参。对 P1D2/P2D1，该式严格退化为已经验证过的 R61 max/min 公式。

## 资源口径

单个 shard 最多使用 **2 个节点、5 个 GPU 进程**。inventory 中 A0–A3/B0–B3 是 GPU 插槽，不是 8 个节点；L1 在一个 8 卡节点内，L2/L3 各在两个节点上。所有 shard、replica 和 topology 顺序执行，禁止把它申请成 4 节点任务。

## 执行顺序

1. 从精确 W64 建立 run 分支，填写 Git 外 `topology_inventory.json`。
2. `make_topology_plan.py` 冻结 plan；此后不得换快卡、挑方向或按结果选 placement。
3. 在 `lmsysorg/sglang:v0.5.15` 中运行 `preflight.py`。
4. 用 `render_launch_commands.py` 为一个 shard/repeat 生成 4 或 5 条需同时启动的命令。
5. 每个 shard 先跑 5 个独立 repeat；`raw_status.py` 仅按 CV 合同机械追加到 7 或 9。
6. raw 完整后运行 `run.py`、`verify.py`，仅按 `commit_allowlist.txt` 提交结果。

Phase64 即使精度门失败也要如实回传；失败标签只能作为未来 Phase65 development，不能在本阶段偷偷拟合。
