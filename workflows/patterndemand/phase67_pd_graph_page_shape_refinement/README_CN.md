# Phase67：PD多流通信图 page-shape 修正开发

## 要解决的问题

R65只从Phase64的page `{1,2,4,8,16}` 学到了 `M/B/S` 图结构公式。Phase66换到page `{3,6,12,24,32}` 后，整体仍达标，但Qwen逐模型WAPE为11.654%，说明“图怎么连”之外，“每条流由多少KV page组成”还会改变并发拥塞。

Phase67不重跑GPU。它把R64和R66共480个已经公开的物理点都作为development数据，比较三档预注册、低容量公式：

1. 原来的 `M/B/S` 仿射式；
2. 加入最大流page数和其余流page数；
3. 再加入两项平方根page特征，表达随消息增大逐渐饱和的效应。

模型固定按 `model × configuration` 分组，不搜索神经网络、不自由造特征、不以训练集误差选模型。

## 防止“调到盲测上”

拟合前已经在 `phase68_reserved_blind_grid.json` 冻结全新的page `{36,40,48,56,64}`。Phase67只能检查该文件与旧page零重叠，禁止读取Phase68测量和target。

候选必须同时通过四道门：20折payload cohort OOF、3折topology OOF、Phase64/66双向source-blocked OOF，以及把含page32样本完全留出的tail外推。每道门都沿用整体/逐模型/逐配置10%，配置×拓扑15%，并且必须优于max-edge、R61和R65三个baseline。

## 执行

```bash
W67=$(git rev-parse HEAD)
python3 workflows/patterndemand/phase67_pd_graph_page_shape_refinement/preflight.py --expected-workflow-commit "$W67"
python3 workflows/patterndemand/phase67_pd_graph_page_shape_refinement/run.py --expected-workflow-commit "$W67"
python3 workflows/patterndemand/phase67_pd_graph_page_shape_refinement/verify.py
```

必须在从W67创建的干净隔离worktree/run分支执行。预计数秒至一分钟；GPU、网络和新物理测量均禁止。

## 结论边界

`PASS`只表示该低容量公式在两批development物理数据上的四种留出检验达标。它还不是fresh-blind结论；只有后续Phase68按已冻结新网格实测并机械验收后，才能判断这次修正是否真正泛化。
