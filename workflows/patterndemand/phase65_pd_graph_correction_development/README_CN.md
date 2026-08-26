# Phase65：PD多流通信图修正开发

Phase64执行有效，但旧R61图扩展公式整体WAPE为19.69%，未达到10%门。Phase65把这240个结果点明确作为development数据，在本地CPU上选择可解释的小公式；不使用GPU，不新增物理数据，也不打开Phase66 target。

## 输入与公式

对每个点定义：

- `M`：最慢单边的Phase51曲线代价；
- `B`：最忙P或D端点相连边的曲线代价和；
- `S`：全部边曲线代价和。

候选统一使用：

```text
T = max(1, intercept + beta_M*M + beta_busy*(B-M) + beta_nonbusy*(S-B))
```

复杂度从全局一套系数，依次增加到graph class、model、model×graph class、model×configuration。只允许选择第一个同时通过payload留出和topology留出合同的候选。

## 精度合同

两种OOF都必须满足：整体、每模型、每配置WAPE≤10%；每配置×拓扑≤15%；signed bias同门；全预测为正；整体优于max-edge和旧R61公式；每个配置优于两者中更好的baseline。

Phase65通过只代表开发交叉验证和公式冻结。正式结论必须等Phase66使用未见page `{3,6,12,24,32}`、新endpoint tuple和未打开target做GPU fresh-blind。

## 本地执行

```bash
python3 workflows/patterndemand/phase65_pd_graph_correction_development/run.py \
  --expected-workflow-commit <W65>

python3 workflows/patterndemand/phase65_pd_graph_correction_development/verify.py
```

必须从干净的精确W65运行；不得覆盖既有结果目录，不得`git add .`。
