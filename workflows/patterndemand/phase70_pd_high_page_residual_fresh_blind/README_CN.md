# Phase70：PD高page残差第三次fresh-blind物理验证

Phase69把R64、R66、R68共720个公开物理点作为development，冻结了一个以R67为底座的低容量高page残差公式。Phase70不训练，只在Phase69拟合前封存的新page和新placement上做正式GPU盲测。

## 冻结预测

每条边先查Phase51单链路曲线，得到`C_e`；再计算`M=max(C_e)`、`B=最忙端点的边代价和`、`S=sum(C_e)`，由冻结R67公式得到`T_R67`。

对P1D4、P4D1和P2D2 all-to-all三种共享端点图：

```text
T_R69 = max(1,
  T_R67
  + gamma_max  * max(0, page_max - 32)
  + gamma_rest * max(0, mean_other_pages - 32))
```

P2D2 matching没有共享端点，`T_R69 == T_R67`。八组`gamma`对应2个模型×4种图配置。Phase70真实时间不得用于改公式、系数、阈值或样本。

## fresh-blind边界

- 只测拟合前封存的page `{34,38,44,52,60}`，与Phase64/66/68的15种development page零重叠；
- 五种page都不是Phase51 knot：payload与descriptor按冻结的每page结构精确线性生成，曲线时间在32–64之间做既定log2插值；
- 每个`(host, physical_gpu, ib_device)`必须同时避开Phase64、Phase66和Phase68 plan；
- 每种L1/L2/L3至少一个host signature也必须是三期都未见的；
- inventory必须在任何Phase70 raw前按scheduler/asset元数据冻结，禁止看速度或预测误差挑placement；
- 48个shard，每个5次起测，CV只能机械追加到7/9次；raw、plan、preflight均在Git外。

## 资源口径

可以一次预约4节点作为placement池，但不是四节点collective。每个shard只激活1或2节点、4或5个GPU进程；48个shard严格顺序执行，禁止两个shard同时占fabric。A0–A3/B0–B3是GPU插槽，不是八个节点。

## 验收

冻结R69必须满足：

- 整体、逐模型、逐配置、逐模型×配置WAPE与signed bias不超过10%；
- 逐配置×拓扑WAPE与signed bias不超过15%；
- 整体严格优于max-edge、R61、R65和R67；
- P1D4、P4D1、P2D2 all-to-all分别优于四个baseline中的最好者；
- P2D2 matching逐点严格保持R67。

失败仍是有效fresh-blind证据，必须保存，禁止在Phase70内调参。

## 命令骨架

`$BASE`必须位于Git外：

```bash
python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/make_topology_plan.py \
  --inventory "$BASE/topology_inventory.json" --output "$BASE/topology_plan.json"

# 容器内设置repo python first、MOONCAKE_PROTOCOL=rdma、
# WITH_NVIDIA_PEERMEM=0、SGLANG_DISAGG_STAGING_BUFFER=0、HF/Transformers offline。
python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/preflight.py \
  --expected-workflow-commit <W70> --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" --audit-output "$BASE/preflight.json" \
  --container-image lmsysorg/sglang:v0.5.15

python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/render_launch_commands.py \
  --topology-plan "$BASE/topology_plan.json" --measurement-id <ID> --repeat-id <N> \
  --raw-dir "$BASE/raw" --master-addr <ADDR> --master-port <PORT>

python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/raw_status.py \
  --topology-plan "$BASE/topology_plan.json" --raw-dir "$BASE/raw"

python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/run.py \
  --expected-workflow-commit <W70> --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" --preflight-audit "$BASE/preflight.json"

python3 workflows/patterndemand/phase70_pd_high_page_residual_fresh_blind/verify.py
```
