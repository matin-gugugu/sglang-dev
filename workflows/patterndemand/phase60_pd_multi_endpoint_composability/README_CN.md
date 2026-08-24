# Phase60：P1D2/P2D1单链路可组合性物理实验

Phase60冻结既有P1D1消息预测链和Phase51物理曲线，不训练、不推理、不加载模型。它只回答：两条KV传输在同一wave内被同时放行后，`max(P1D1曲线耗时)`能否预测整个wave的完成时间。

正式矩阵为2个代表模型（Qwen3-8B与DeepSeek-V2-Lite）×2个配置（P1D2 fan-out、P2D1 fan-in）×L1/L2/L3×2套预先冻结placement，共24个三rank measurement shard。每个shard只测合同中的10个development payload pair，并同时测`solo_flow0`、`solo_flow1`和`concurrent_wave`。未来blind pair明确保留，Phase60禁止测量。

P1D2中一个P rank通过两个worker thread在同一生产`MooncakeTransferEngine`、两个互不重叠注册区上向两个D peer调用`batch_transfer_sync`；P2D1中两个P rank分别向一个D rank调用同一生产原语。整wave先做跨rank屏障，再放行两路调用。若生产引擎实际串行，这也是应保留的真实结果；禁止替换wrapper/backend来制造重叠。

## 为什么同时测solo锚点

Phase60同时计算两种baseline：

1. 冻结Phase51曲线：`max(C(payload0), C(payload1))`；
2. 同一shard、同一环境的matched solo：`max(solo0, solo1)`。

若两者都无法解释并发wave，才有直接的contention证据。若旧Phase51曲线失败、matched solo通过，则更可能是跨日期/placement环境漂移，不能把它错误写成contention。

## 执行顺序

在exact W60上创建唯一run分支。将`topology_inventory.example.json`复制到Git外，填写6套placement。每套placement冻结A侧2个GPU endpoint和B侧2个GPU endpoint；每个实际shard使用其中3个endpoint。L1四个slot同机，L2两侧主机同rack，L3两侧主机跨rack。

```bash
P60=workflows/patterndemand/phase60_pd_multi_endpoint_composability
EXT=/EXTERNAL/phase60_attempt1
mkdir -p "$EXT"
cp "$P60/topology_inventory.example.json" "$EXT/topology_inventory.json"
# 填写真实host/rack/fabric/GPU/HCA证据后：
python3 "$P60/make_topology_plan.py" \
  --inventory "$EXT/topology_inventory.json" \
  --output "$EXT/topology_plan.json"
```

冻结离线RDMA环境并运行preflight：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MOONCAKE_PROTOCOL=rdma WITH_NVIDIA_PEERMEM=0 SGLANG_DISAGG_STAGING_BUFFER=0
unset MC_FORCE_TCP MC_FORCE_MNNVL MC_INTRANODE_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL

python3 "$P60/preflight.py" \
  --expected-workflow-commit W60 \
  --container-image lmsysorg/sglang:v0.5.15 \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$EXT/raw" \
  --audit-output "$EXT/preflight.json"
```

对24个measurement分别运行repeat 0–4。渲染器返回3条逐rank命令；必须在对应主机上并发启动，不能顺序执行：

```bash
python3 "$P60/render_launch_commands.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --measurement-id qwen3-8b__p1d2__l1__r0 \
  --repeat-id 0 --raw-dir "$EXT/raw" \
  --master-addr HOST0_REACHABLE_ADDR --master-port 29660
```

每完成一轮运行：

```bash
python3 "$P60/raw_status.py" --topology-plan "$EXT/topology_plan.json" --raw-dir "$EXT/raw"
```

只对`needs_extra`列出的measurement追加repeat 5–6；仍高方差再追加7–8。不得删异常、挑快placement或降低负载。完整后：

```bash
python3 "$P60/run.py" \
  --expected-workflow-commit W60 \
  --topology-plan "$EXT/topology_plan.json" --raw-dir "$EXT/raw" \
  --preflight-audit "$EXT/preflight.json"
python3 "$P60/verify.py"
```

只选择性添加`experiment-results/phase60_pd_multi_endpoint_composability/`并运行`verify_staging.py --phase phase60`。raw、模型、缓存、PID和密钥不得进入Git；禁止`git add .`。

## Phase60如何下结论

- Phase51 baseline总体WAPE≤10%且每个`配置×拓扑`≤15%：`P1D1_DIRECTLY_COMPOSABLE_DEVELOPMENT`；
- matched-solo baseline超过上述阈值：`CONTENTION_CORRECTION_CANDIDATE`；
- Phase51失败但matched-solo通过：`P1D1_CURVE_TRANSFER_DRIFT_REQUIRES_REVIEW`。

这些都是development物理证据。Phase60不拟合修正项，也不打开未来blind pair。下一阶段只有在contention证据成立时，才用development数据拟合轻量倍率并冻结后续blind workflow。
