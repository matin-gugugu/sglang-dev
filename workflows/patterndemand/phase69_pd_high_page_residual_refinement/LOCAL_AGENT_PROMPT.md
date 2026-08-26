# Phase69本地执行Agent提示词

从正式分支W69创建独立干净run分支，完整阅读本目录全部合同文件，只执行README_CN.md中的preflight、run和verify。不得使用GPU或网络，不得生成新物理数据，不得访问任何Phase70 measurement/target，不得修改候选、阈值、fold或输出。

verify通过后，严格按`commit_allowlist.txt`逐文件选择性添加结果，禁止`git add .`。结果commit必须只有W69一个父提交；push run分支并报告W69、R69、状态、四种validation指标、选中公式、模型×配置最差指标、结果目录、manifest和结论边界。
