# Phase61：P1D2/P2D1并发通信修正

Phase60已经证明：两条P-D链路同时运行时，直接取两条P1D1曲线的最大值会产生约28%误差。Phase61不使用GPU，只用Phase60已经冻结的120个development物理点，寻找一个足够简单的并发修正公式。

输入只有调度器实际能获得的两条Phase51单链路曲线值C0/C1。matched_solo和真实并发时间只作为development监督标签，不能成为最终预测输入。

候选由简单到复杂依次为统一倍率，以及：

~~~text
预测并发时间 =
    intercept
  + beta_max × max(C0,C1)
  + beta_min × min(C0,C1)
~~~

选择使用20折leave-one-payload-pair-out：每次把一个payload pair在P1D2/P2D1和L1/L2/L3下的六行全部留出，避免同一payload在训练和验证中泄漏。第一个达到整体WAPE≤10%、每个“配置×拓扑”WAPE≤15%及对应bias阈值的候选被选中。达到阈值后禁止为了更小development误差选择更复杂模型。

运行：

~~~bash
P61=workflows/patterndemand/phase61_pd_contention_correction
python3 "$P61/preflight.py" --expected-workflow-commit W61
python3 "$P61/run.py" --expected-workflow-commit W61
python3 "$P61/verify.py"
~~~

Phase61不打开reserved future blind、不运行GPU、不重新测物理曲线。只有Phase61冻结模型后，Phase62才允许在GPU上测预留payload和未见placement。
