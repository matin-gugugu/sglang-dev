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

推荐在调度器中一次保留 **4 个节点**，因为目标环境若更容易排到四节点整组，就不必为每个 topology 反复排队。但“保留4节点”不等于“四节点同时通信”：单个 shard 仍最多激活 **2 个节点、5 个 GPU 进程**，其余已保留节点保持空闲。inventory 中 A0–A3/B0–B3 是 GPU 插槽，不是节点。

L1 只在一个节点内测；L2/L3 的一组通信只跨两个节点。模型、配置、topology 与 replica 的 raw 测量彼此没有计算依赖，可以换序；只有 5 次初测后的 CV 决定是否追加到 7/9 次，最终聚合依赖全部 raw。禁止同时跑两个 shard，因为它们会互相抢 fabric，污染正在测量的通信代价。

## 执行顺序

1. 从精确 W64 建立 run 分支，填写 Git 外 `topology_inventory.json`。
2. 在 inventory 中选择 `FOUR_NODE_SINGLE_ALLOCATION`（推荐）或 `SEQUENTIAL_TOPOLOGY_EPOCHS`，再由 `make_topology_plan.py` 冻结 plan；此后不得换快卡、挑方向或按结果选 placement。
3. 在 `lmsysorg/sglang:v0.5.15` 中运行 `preflight.py`。
4. 用 `render_launch_commands.py` 为一个 shard/repeat 生成 4 或 5 条需同时启动的命令。
5. 每个 shard 先跑 5 个独立 repeat；`raw_status.py` 仅按 CV 合同机械追加到 7 或 9。
6. raw 完整后运行 `run.py`、`verify.py`，仅按 `commit_allowlist.txt` 提交结果。

Phase64 即使精度门失败也要如实回传；失败标签只能作为未来 Phase65 development，不能在本阶段偷偷拟合。
