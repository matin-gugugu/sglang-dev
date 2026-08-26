# Phase65 本地执行Agent合同

从控制端指定的精确W65运行。完整阅读本目录全部文件。Phase64的240个official point是唯一训练/选择标签；GPU、网络、新物理测量和Phase66 target全部禁止。

先运行单测和preflight，再运行`run.py`与`verify.py`。候选必须按固定复杂度顺序机械选择，不能为了更低refit误差跳过已经通过的简单候选，不能降低10%/15%门。

结果只能按`commit_allowlist.txt`选择性提交，唯一父提交必须为W65。报告W65/R65、运行时间、候选顺序、两个OOF协议、基线与选中公式误差、是否允许Phase66、结果目录、summary/README/log/DONE/manifest，以及不能得出的fresh-blind结论。
