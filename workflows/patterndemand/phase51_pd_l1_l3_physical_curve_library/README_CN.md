# Phase51 workflow：纯PD L1–L3物理通信曲线库

Phase50已经完成六模型纯PD消息直方图预测闭环；Phase51不再训练预测器，也不加载模型。它只回答下一层问题：一个P实例要把某段KV发给D实例时，在冻结的L1/L2/L3 placement上实际需要多长时间。

这里不能把逻辑KV chunk简化为一次普通`memcpy`。正式测量直接调用SGLang生产源码中的`MooncakeTransferEngine.batch_transfer_sync`：五个MHA/GQA模型按`2 × layer`个K/V描述符传输，DeepSeek MLA按`layer`个描述符传输。每次调用的总字节数与Phase48冻结的模型结构公式严格一致。

## 实验规模与保守汇总

- 6个模型 × L1/L2/L3 × 每类2个预先冻结的placement = 36个measurement shard；
- 每个shard覆盖对应模型的完整payload网格，并测两个方向；
- 每个shard先做5次独立重复。任一payload/方向的重复中位数CV超过15%，只追加到7次，再按相同规则最多追加到9次；不得删除高值；
- 每个placement的正式点取两个方向中较慢者；每条曲线再取两个placement中较慢者；
- 最终是18条模型相关物理曲线、396个knots。逐iteration raw JSONL永久留在Git外，Git只保存紧凑曲线、环境/拓扑/方差审计与raw哈希。

L1是同机两张不同GPU，但仍强制Mooncake RDMA；L2是同rack不同主机；L3是跨rack但同一声明RDMA fabric。L1/L2/L3来自测量前的机器/rack/fabric元数据，绝不能根据跑出来的快慢倒推或改标签。Phase51允许真实排序不是L1<L2<L3。

## 远程执行顺序

在exact W51创建唯一run分支，完整阅读本目录所有合同和脚本。先从调度系统/机器资产元数据确认6个placement，把`topology_inventory.example.json`复制到Git外并替换全部占位符。两个replica必须是不同endpoint组合；在第一条benchmark raw产生前冻结：

多机shard要求仓库和`EXT`都位于参与主机可用的共享文件系统，并在所有主机上解析为相同绝对路径；raw仍然必须位于Git仓库之外。

```bash
P51=workflows/patterndemand/phase51_pd_l1_l3_physical_curve_library
EXT=/EXTERNAL/phase51_attempt1
mkdir -p "$EXT"
cp "$P51/topology_inventory.example.json" "$EXT/topology_inventory.json"
# 编辑并复核 $EXT/topology_inventory.json；不能用速度测试决定L1/L2/L3
python3 "$P51/make_topology_plan.py" \
  --inventory "$EXT/topology_inventory.json" \
  --output "$EXT/topology_plan.json"
```

正式环境固定为仓库源码优先、离线、RDMA/dma-buf且无fallback。preflight只审计环境并创建空raw根，不运行传输：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MOONCAKE_PROTOCOL=rdma WITH_NVIDIA_PEERMEM=0 SGLANG_DISAGG_STAGING_BUFFER=0
unset MC_FORCE_TCP MC_FORCE_MNNVL MC_INTRANODE_NVLINK SGLANG_MOONCAKE_CUSTOM_MEM_POOL

python3 "$P51/preflight.py" \
  --expected-workflow-commit W51 \
  --container-image lmsysorg/sglang:v0.5.15 \
  --topology-plan "$EXT/topology_plan.json" \
  --raw-dir "$EXT/raw" \
  --audit-output "$EXT/preflight.json"
```

对36个`measurement_id`分别运行repeat 0–4。以下命令只渲染精确argv；执行Agent要为本次repeat选择未占用的控制端口，将JSON中的一条（L1）或两条（L2/L3）命令在对应主机并发启动。不同placement只有在GPU/NIC互不冲突时才可并行：

```bash
python3 "$P51/render_launch_commands.py" \
  --topology-plan "$EXT/topology_plan.json" \
  --measurement-id qwen3-8b__l1__r0 --repeat-id 0 \
  --raw-dir "$EXT/raw" --master-addr HOST0_REACHABLE_ADDR --master-port 29651
```

每完成一轮运行：

```bash
python3 "$P51/raw_status.py" --topology-plan "$EXT/topology_plan.json" --raw-dir "$EXT/raw"
```

只对`needs_extra`列出的measurement追加repeat 5–6，再次检查；仍在列表中的再追加7–8。不得把额外重复只加给“看起来更好”的方向，benchmark会对整个shard的全部payload和双向重新测量。`complete=true`后才允许聚合：

```bash
python3 "$P51/run.py" \
  --expected-workflow-commit W51 \
  --topology-plan "$EXT/topology_plan.json" --raw-dir "$EXT/raw" \
  --preflight-audit "$EXT/preflight.json"
python3 "$P51/verify.py"
```

只选择性添加`experiment-results/phase51_pd_l1_l3_physical_curve_library/`。被Git忽略的allowlist文件逐个`git add -f`，禁止`git add .`。运行`verify_staging.py --phase phase51`后形成唯一父提交为W51的单一R51并push run分支。

## 可以适应与不可改变

`AUTO/RECORD_AND_CONTINUE`允许：在生成plan以前按元数据挑空闲的同类GPU/HCA；诊断端口/hostname/IB可达性；重试失败shard并保留旧attempt；让互不冲突的placement并行；按固定CV规则追加重复。plan冻结后若必须换endpoint，整个attempt作废，复制到新目录、重新冻结并从repeat 0开始。

禁止：TCP/MNNVL/NVLink/staging/custom-pool回退；挑更快方向或replica；删除异常；降低payload网格或重复数；改模型描述符；把raw、模型、缓存、PID、密钥加入Git；运行Phase52卷积/placement；触碰任何历史保护目录。

Phase51只证明冻结环境上的物理通信曲线，不证明端到端请求延迟、计算时间、显存可行性、排队/拥塞、资源空闲、通信计算重叠或placement最优性。
曲线还明确假设每层对应的物理page index构成一个连续区间；任意内存碎片造成的多区间描述符膨胀不在本阶段结论内。
