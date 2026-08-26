# Phase68：PD多流 graph+page-shape 第二次 fresh-blind 物理验证

Phase67把Phase64与Phase66共480个公开物理点作为development，冻结了一个低容量`model×configuration`公式。Phase68不训练，只在全新page和placement上做第二次正式盲测。

## 冻结预测

每条边先查Phase51单链路曲线，得到`C_e`；`M=max(C_e)`、`B=最忙端点的边代价和`、`S=sum(C_e)`。page向量再产生`pmax=max(pages)`和`prest=sum(pages)-pmax`：

```text
T_hat = max(1,
  beta0 + betaM*M + betaB*(B-M) + betaS*(S-B)
  + betaPmax*pmax + betaPrest*prest
  + betaQmax*sqrt(pmax) + betaQrest*sqrt(prest))
```

八组冻结系数对应2个模型×4种图配置。Phase68真实时间不得用于改公式、系数、阈值或样本。

## fresh-blind边界

- 只测拟合前封存的page `{36,40,48,56,64}`，与Phase64/66的十种page完全零重叠；
- 36/40/56不是Phase51 knot：payload与descriptor按冻结的每page结构精确线性生成，曲线时间在32–48或48–64之间做既定log2插值；
- 每个`(host, physical_gpu, ib_device)`必须同时避开Phase64和Phase66 plan；
- 每种L1/L2/L3至少一个host signature也必须是两期都未见的；
- inventory必须在任何Phase68 raw前按scheduler/asset元数据冻结，禁止看速度挑placement；
- 48个shard，每个5次起测，CV只能机械追加到7/9次；raw、plan、preflight均在Git外。

## 资源口径

可以一次预约4节点作为placement池，但不是四节点collective。每个shard只激活1或2节点、4或5个GPU进程，并且48个shard严格顺序执行，禁止两个shard同时占fabric。

## 验收

冻结R67必须满足整体/逐模型/逐配置WAPE和signed bias≤10%，逐配置×拓扑≤15%；整体严格优于max-edge、R61、R65三个baseline，每个配置也严格优于三者中的最好者。失败仍是有效fresh-blind证据，必须保存且禁止原地调参。

## 命令骨架

`$BASE`必须位于Git外：

```bash
python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/make_topology_plan.py \
  --inventory "$BASE/topology_inventory.json" --output "$BASE/topology_plan.json"

# 容器内先设置：repo python first、MOONCAKE_PROTOCOL=rdma、
# WITH_NVIDIA_PEERMEM=0、SGLANG_DISAGG_STAGING_BUFFER=0、HF/Transformers offline。
python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/preflight.py \
  --expected-workflow-commit <W68> --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" --audit-output "$BASE/preflight.json" \
  --container-image lmsysorg/sglang:v0.5.15

python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/render_launch_commands.py \
  --topology-plan "$BASE/topology_plan.json" --measurement-id <ID> --repeat-id <N> \
  --raw-dir "$BASE/raw" --master-addr <ADDR> --master-port <PORT>

python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/raw_status.py \
  --topology-plan "$BASE/topology_plan.json" --raw-dir "$BASE/raw"

python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/run.py \
  --expected-workflow-commit <W68> --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" --preflight-audit "$BASE/preflight.json"

python3 workflows/patterndemand/phase68_pd_graph_page_shape_fresh_blind/verify.py
```
