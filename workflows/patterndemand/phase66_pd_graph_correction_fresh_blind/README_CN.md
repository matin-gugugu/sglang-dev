# Phase66：PD多流图修正 fresh-blind 物理验证

Phase65用Phase64的240点开发数据冻结了一个`model×configuration`轻量公式；Phase66不再训练，而是用预先封存、从未测过的page和全新GPU endpoint做正式盲测。

## 冻结对象

每条边先查Phase51单链路曲线，得到`C_e`。令`M=max(C_e)`、`B=最忙端点相连边代价和`、`S=sum(C_e)`，R65公式为：

```text
T_hat = max(1, intercept + beta_M*M + beta_busy*(B-M) + beta_nonbusy*(S-B))
```

八组系数分别对应两个模型×四种配置。不得用Phase66的真实时间重拟合、改系数或改门槛。

## fresh-blind 边界

- 只测封存page `{3,6,12,24,32}`，与Phase64 `{1,2,4,8,16}` 零重叠；
- inventory内每个`(host, physical_gpu, ib_device)`都不得出现在Phase64 plan；
- 每种L1/L2/L3至少一个host signature也必须是新的；
- placement必须在任何Phase66 raw产生前按调度器元数据冻结，不能看速度挑卡；
- 48个shard、每个5次起测，CV只允许机械追加到7/9次；raw只在Git外。

## 资源口径

允许一次预约4节点以便排队和固定placement池，但单个shard只激活1或2节点、4或5个GPU进程；同一时刻只能跑1个shard。它不是四节点collective，也不能把两个两节点shard并发执行。

## 验收

R65预测必须满足：整体/每模型/每配置WAPE与signed bias≤10%，每配置×拓扑≤15%，并且整体优于max-edge和旧R61，且每配置优于两者中更好的baseline。失败也必须完整保存为fresh-blind证据，禁止原地调参。

执行Agent从精确W66建立run分支，依次执行`make_topology_plan.py`、容器内`preflight.py`、GPU shard、`raw_status.py`、`run.py`、`verify.py`；只按`commit_allowlist.txt`提交紧凑结果。

## 命令骨架

以下`$BASE`必须在Git仓库外，plan、preflight和raw不能进Git：

```bash
python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/make_topology_plan.py \
  --inventory "$BASE/topology_inventory.json" \
  --output "$BASE/topology_plan.json"

# 在 lmsysorg/sglang:v0.5.15 内，并设置repo python first、RDMA/dma-buf、离线模型环境：
python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/preflight.py \
  --expected-workflow-commit <W66> \
  --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" \
  --audit-output "$BASE/preflight.json" \
  --container-image lmsysorg/sglang:v0.5.15

# 每个measurement/repeat先渲染并在指定主机并发启动该shard的4或5条rank命令：
python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/render_launch_commands.py \
  --topology-plan "$BASE/topology_plan.json" --measurement-id <ID> --repeat-id <N> \
  --raw-dir "$BASE/raw" --master-addr <ADDR> --master-port <PORT>

python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/raw_status.py \
  --topology-plan "$BASE/topology_plan.json" --raw-dir "$BASE/raw"

python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/run.py \
  --expected-workflow-commit <W66> --topology-plan "$BASE/topology_plan.json" \
  --raw-dir "$BASE/raw" --preflight-audit "$BASE/preflight.json"

python3 workflows/patterndemand/phase66_pd_graph_correction_fresh_blind/verify.py
```
