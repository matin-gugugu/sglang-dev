# Phase69：PD大page残差修正开发

## 这一步做什么

Phase68证明R67整体误差只有1.62%，但Qwen3-8B在大page的P1D4和P2D2 all-to-all上持续高估，其中all-to-all逐模型配置WAPE为13.09%。Phase69不重跑GPU，而是把R64、R66、R68共720个物理点作为development数据。

新公式不推翻R67。它把R67当成固定底座，只对存在共享端点竞争的P1D4、P4D1和P2D2 all-to-all，在page超过32时学习“最大流超过32多少页”和“其余流平均超过32多少页”两项残差。page不超过32时修正严格为零；没有共享端点的P2D2 matching在所有page上也保持R67不变。

候选固定为线性、平方根、线性加平方根三档；按复杂度从低到高选择第一个通过合同的候选，不搜索神经网络、不自由造特征。

## 四道检验

1. 30折source×payload cohort留出；
2. 3折topology留出；
3. 完整留出所有含page64的尾部样本；
4. Phase64/66/68逐批source-blocked诊断。

前三道必须在整体以及三种共享端点通信形态上优于max-edge、R61、R65、R67；P2D2 matching必须严格保持R67。每个“模型×通信形态”另外必须满足10%精度门，但不强迫已经接近零误差的切片继续严格变小。第四道允许在没有大page训练证据时与R67相等，但不能退化。精度门为整体、逐模型、逐配置和逐模型×配置10%，更细的配置×拓扑与source×模型×配置为15%。

## Phase70盲测边界

拟合前已冻结Phase70 page `{34,38,44,52,60}`，与R64/R66/R68全部development page零重叠，且要求新的endpoint tuple。Phase69不得读取任何Phase70测量或target。

## 执行

```bash
W69=$(git rev-parse HEAD)
python3 workflows/patterndemand/phase69_pd_high_page_residual_refinement/preflight.py --expected-workflow-commit "$W69"
python3 workflows/patterndemand/phase69_pd_high_page_residual_refinement/run.py --expected-workflow-commit "$W69"
python3 workflows/patterndemand/phase69_pd_high_page_residual_refinement/verify.py
```

必须在W69创建的干净隔离worktree/run分支执行。预计数秒至一分钟；GPU、网络和新物理测量均禁止。

## 结论边界

`PASS`只表示高page修正在三批公开development数据的预注册留出检验中达标。它不是fresh-blind结论；只有Phase70使用冻结网格完成新GPU实测后，才能判断修正是否真正泛化。
